"""Fact-checker nodes for the analyst stage (CODE-ONLY, SELF-CHECK).

After fundamentals / technical / game_theory produce their report, the
fact-checker node verifies the claims **in code** — no LLM is involved and
**no new data request is made**:

1. It reads the analyst's OWN tool data from the FIRST request, which is
   kept verbatim in the analyst branch's conversation (ToolMessage in
   messages_<analyst>) — 自己检查自己的。The tool is therefore called
   exactly ONCE per analyst per run (analyst fetch only); retries reuse the
   same context. A re-fetch fallback exists only for anomalous flows where
   the tool evidence is missing.
2. Every claim is checked deterministically against that JSON:
   - fact: code resolves ``evidence[].json_path`` and compares the exact field
     value with the report value;
   - calculation: code resolves every formula variable from its evidence path,
     re-evaluates the expression, and compares it with the report value.
   The LLM never provides a trusted copy of the raw value.
3. On failure the code-generated feedback (which claims failed and why) is
   stored in verification_state; the subgraph router sends the analyst back
   to redo its material.  The retry keeps the FIRST tool round's context and
   binds no tools, so no new data request is made on retry (bounded by
   max_verify_rounds).  Beyond the budget the report is marked unverified
   and the pipeline continues.
"""

from __future__ import annotations

import ast
import json
import logging
import math
import re
from typing import Any, Callable, Dict, List, Optional

from tradingagents.agents.schemas import (
    FactVerificationItem,
    FactVerificationReport,
    render_verification_report,
)
from tradingagents.errors import AnalysisStopped
from langchain_core.messages import ToolMessage
from tradingagents.agents.utils.external_api_tools import (
    tool_fundamental,
    tool_game_theory,
    # tool_news_sentiment,  # 交叉校验已去掉（2026-08：自己检查自己的）
    # tool_risk,  # 原 /risk 端点已停用（404），tool_risk 已删除（2026-08）
    # tool_special_data,  # 外部 API 无此接口，已停用（2026-08）
    tool_technical,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Verification tool set (SELF-CHECK only).
#
# 校验只检查"自己"：每个分析师只复拉并校验**自身工具**的数据（来源是否属实、
# 计算是否正确）。不做跨源比对（2026-08：交叉校验先去掉，自己检查自己的即可）。
# ---------------------------------------------------------------------------
VERIFY_TOOLS: Dict[str, List[Any]] = {
    "fundamentals": [tool_fundamental],
    "technical": [tool_technical],
    "game_theory": [tool_game_theory],
}

VERIFY_ANALYSTS = ("fundamentals", "technical", "game_theory")

_REPORT_KEYS = {
    "fundamentals": "fundamentals_report",
    "technical": "technical_report",
    "game_theory": "game_theory_report",
}

_CLAIM_KEYS = {
    "fundamentals": "fundamentals_claims",
    "technical": "technical_claims",
    "game_theory": "game_theory_claims",
}

# 相对误差容差：报告值与代码复算值偏差超过 0.5% 视为不一致
_REL_TOLERANCE = 0.005

# 数值单位换算（A股报告常见量级）
_UNITS = {"%": 0.01, "万": 1e4, "亿": 1e8, "万亿": 1e12}
_PERCENT_UNITS = {"%", "percent", "percentage", "pct", "百分比", "比例"}
_MISSING = object()
_EVAL_ERROR = object()

# LLM-generated expressions are data, not programs.  Bound their size and
# every intermediate numeric result so a malformed expression cannot turn a
# fact-check into an expensive computation (for example 10 ** (10 ** 10)).
_MAX_EXPRESSION_CHARS = 500
_MAX_EXPRESSION_NODES = 100
_MAX_EXPRESSION_DEPTH = 20
_MAX_AGGREGATE_ITEMS = 10_000
_MAX_POWER_EXPONENT = 1_000.0
_MAX_ABS_NUMBER = 1e300


# ---------------------------------------------------------------------------
# Code-level verification helpers
# ---------------------------------------------------------------------------


def _norm(text: Any) -> str:
    """Collapse all whitespace so verbatim excerpts match re-serialized JSON."""
    return "".join(str(text).split())


def _parse_num(value: str) -> tuple[Optional[float], bool]:
    """Parse a report value like '33.05%' / '12.3亿' / '500万股' -> (number, has_percent)."""
    text = str(value or "").strip()
    m = re.search(r"(-?(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?)\s*(万亿|亿|万|%)?", text)
    if not m:
        return None, False
    num = float(m.group(1).replace(",", ""))
    unit = m.group(2) or ""
    has_percent = unit == "%"
    num *= _UNITS.get(unit, 1.0)
    return num, has_percent


def _close(a: float, b: float) -> bool:
    """Relative tolerance comparison (guards against float noise)."""
    if not math.isfinite(a) or not math.isfinite(b):
        return False
    if b == 0:
        return abs(a) < 1e-9
    return abs(a - b) <= max(1e-9, abs(b) * _REL_TOLERANCE)


def _json_path_parts(path: str) -> Optional[List[Any]]:
    """Parse a small, deterministic JSONPath/JSON-Pointer subset.

    Supported examples: ``$.data.metrics.roic``, ``$.rows[0].close``,
    ``/data/metrics/roic`` and array mapping ``$.rows[*].close`` (the latter
    resolves to a LIST, which formula/rule aggregate functions consume).
    Calls, filters and wildcards over keys are intentionally unsupported:
    an evidence reference must resolve to either one exact value or one
    homogeneous list.
    """
    value = str(path or "").strip()
    if value.startswith("/"):
        return [
            int(part) if part.isdigit() else part.replace("~1", "/").replace("~0", "~")
            for part in value.split("/")[1:]
        ]
    if value == "$":
        return []
    if not value.startswith("$"):
        return None

    parts: List[Any] = []
    i = 1
    while i < len(value):
        if value[i] == ".":
            i += 1
            start = i
            while i < len(value) and value[i] not in ".[":
                i += 1
            if start == i:
                return None
            parts.append(value[start:i])
            continue
        if value[i] == "[":
            end = value.find("]", i + 1)
            if end < 0:
                return None
            token = value[i + 1:end].strip()
            if token == "*":
                parts.append("*")
            elif token.isdigit():
                parts.append(int(token))
            elif len(token) >= 2 and token[0] == token[-1] and token[0] in "\"'":
                parts.append(token[1:-1])
            else:
                return None
            i = end + 1
            continue
        return None
    return parts


def _resolve_parts(value: Any, parts: List[Any]) -> Any:
    """Walk a resolved-JSON path; ``*`` maps the remaining path over a list."""
    if not parts:
        return value
    part = parts[0]
    rest = parts[1:]
    if part == "*":
        if not isinstance(value, list):
            return _MISSING
        out = []
        for item in value:
            resolved = _resolve_parts(item, rest)
            if resolved is _MISSING:
                return _MISSING  # 任一元素缺字段 → 整体视为缺失（严格）
            out.append(resolved)
        return out
    if isinstance(part, int):
        if not isinstance(value, list) or part >= len(value):
            return _MISSING
        return _resolve_parts(value[part], rest)
    if not isinstance(value, dict) or part not in value:
        return _MISSING
    return _resolve_parts(value[part], rest)


def _resolve_json_path(payload: Any, path: str) -> Any:
    parts = _json_path_parts(path)
    if parts is None:
        return _MISSING
    return _resolve_parts(payload, parts)


def _coerce_evidence_number(value: Any, unit: str = "") -> Optional[float]:
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        number = float(value)
        if str(unit or "").strip().lower() in _PERCENT_UNITS:
            number *= 0.01
        return number if math.isfinite(number) else None
    # Never scrape the first number out of a serialized collection.  Array
    # evidence is valid only through an explicit aggregate formula/rule.
    if isinstance(value, (dict, list, tuple, set)) or value is None:
        return None
    number, has_percent = _parse_num(str(value))
    if number is None:
        return None
    if not has_percent and str(unit or "").strip().lower() in _PERCENT_UNITS:
        number *= 0.01
    return number if math.isfinite(number) else None


def _values_match(reported: str, expected: Any, unit: str = "") -> bool:
    """Compare a report value with one resolved raw field, including % scale."""
    if isinstance(expected, (dict, list, tuple, set)):
        return False
    reported_num, reported_percent = _parse_num(reported)
    expected_num = _coerce_evidence_number(expected, unit)
    if reported_num is not None and expected_num is not None:
        if _close(reported_num, expected_num):
            return True
        # Backward-compatible percentage payloads often store 6.0 for 6%.
        if reported_percent and str(unit or "").strip().lower() not in _PERCENT_UNITS:
            return _close(reported_num * 100, expected_num)
        return False
    return _norm(reported) == _norm(expected) or _norm(reported) in _norm(expected)


def _payload_map(raw_data: Dict[str, str]) -> Dict[str, tuple[str, Any]]:
    payloads: Dict[str, tuple[str, Any]] = {}
    for tool_name, data in raw_data.items():
        parsed = None
        try:
            parsed = json.loads(data)
        except Exception:  # noqa: BLE001 - legacy text payload
            pass
        payloads[tool_name] = (str(data or ""), parsed)
    return payloads


def _resolve_evidence(evidence: dict, payloads: Dict[str, tuple[str, Any]]) -> tuple[Any, str]:
    tool_name = str(evidence.get("source_tool") or "")
    path = str(evidence.get("json_path") or "")
    payload = payloads.get(tool_name)
    if payload is None:
        return _MISSING, f"来源工具 {tool_name!r} 不在本次原始数据中"
    if payload[1] is None:
        return _MISSING, f"来源工具 {tool_name!r} 返回的不是结构化 JSON，无法解析路径 {path!r}"
    value = _resolve_json_path(payload[1], path)
    if value is _MISSING:
        return _MISSING, f"JSON 路径 {path!r} 在 {tool_name} 原始数据中不存在"
    return value, ""


def _resolved_evidence(ref: dict, value: Any) -> dict:
    if isinstance(value, (dict, list)):
        display = json.dumps(value, ensure_ascii=False, separators=(",", ":"))
    else:
        display = str(value)
    return {
        "alias": str(ref.get("alias") or ""),
        "source_tool": str(ref.get("source_tool") or ""),
        "json_path": str(ref.get("json_path") or ""),
        "resolved_value": display[:200],
        "unit": ref.get("unit"),
        "period": ref.get("period"),
    }


def _evidence_tools(refs: List[dict]) -> str:
    return ", ".join(dict.fromkeys(str(ref.get("source_tool") or "") for ref in refs if ref.get("source_tool")))


def _expression_text(expression: str, *, allow_assignment_label: bool = False) -> str:
    text = str(expression or "").strip()
    if allow_assignment_label and "=" in text:
        text = text.split("=", 1)[1].strip()
    return text


def _parse_expression(expression: str, *, allow_assignment_label: bool = False) -> Optional[ast.Expression]:
    """Parse one bounded expression, returning ``None`` for unsafe input."""
    text = _expression_text(expression, allow_assignment_label=allow_assignment_label)
    if not text or len(text) > _MAX_EXPRESSION_CHARS:
        return None
    try:
        tree = ast.parse(text, mode="eval")
    except (SyntaxError, RecursionError, ValueError):
        return None
    if sum(1 for _ in ast.walk(tree)) > _MAX_EXPRESSION_NODES:
        return None
    return tree


def _expression_names(expression: str, *, allow_assignment_label: bool = False) -> Optional[set[str]]:
    """变量名集合（排除 sum/avg/median 等被调用的函数名）。"""
    tree = _parse_expression(expression, allow_assignment_label=allow_assignment_label)
    if tree is None:
        return None
    call_funcs = {
        node.func.id
        for node in ast.walk(tree)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
    }
    return {
        node.id
        for node in ast.walk(tree)
        if isinstance(node, ast.Name) and node.id not in call_funcs
    }


def _formula_names(formula: str) -> Optional[set[str]]:
    return _expression_names(formula, allow_assignment_label=True)


def _finite_number(value: Any) -> Optional[float]:
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        return None
    number = float(value)
    if not math.isfinite(number) or abs(number) > _MAX_ABS_NUMBER:
        return None
    return number


def _eval_expression_node(node: ast.AST, env: dict, depth: int = 0) -> Any:
    """Interpret the whitelisted AST without executing generated Python."""
    if depth > _MAX_EXPRESSION_DEPTH:
        return _EVAL_ERROR
    next_depth = depth + 1

    if isinstance(node, ast.Expression):
        return _eval_expression_node(node.body, env, next_depth)
    if isinstance(node, ast.Constant):
        if node.value is None or isinstance(node.value, (str, bool)):
            return node.value
        number = _finite_number(node.value)
        return number if number is not None else _EVAL_ERROR
    if isinstance(node, ast.Name):
        if node.id not in env:
            return _EVAL_ERROR
        value = env[node.id]
        if isinstance(value, list):
            if len(value) > _MAX_AGGREGATE_ITEMS:
                return _EVAL_ERROR
            normalized = []
            for item in value:
                if isinstance(item, (str, bool)) or item is None:
                    normalized.append(item)
                    continue
                number = _finite_number(item)
                if number is None:
                    return _EVAL_ERROR
                normalized.append(number)
            return normalized
        if value is None or isinstance(value, (str, bool)):
            return value
        number = _finite_number(value)
        return number if number is not None else _EVAL_ERROR
    if isinstance(node, ast.List):
        values = [_eval_expression_node(item, env, next_depth) for item in node.elts]
        return _EVAL_ERROR if any(value is _EVAL_ERROR for value in values) else values
    if isinstance(node, ast.UnaryOp):
        operand = _eval_expression_node(node.operand, env, next_depth)
        if operand is _EVAL_ERROR:
            return _EVAL_ERROR
        if isinstance(node.op, ast.Not):
            return not operand if isinstance(operand, bool) else _EVAL_ERROR
        number = _finite_number(operand)
        if number is None:
            return _EVAL_ERROR
        return number if isinstance(node.op, ast.UAdd) else -number
    if isinstance(node, ast.BinOp):
        left = _finite_number(_eval_expression_node(node.left, env, next_depth))
        right = _finite_number(_eval_expression_node(node.right, env, next_depth))
        if left is None or right is None:
            return _EVAL_ERROR
        try:
            if isinstance(node.op, ast.Add):
                result = left + right
            elif isinstance(node.op, ast.Sub):
                result = left - right
            elif isinstance(node.op, ast.Mult):
                result = left * right
            elif isinstance(node.op, ast.Div):
                result = left / right
            elif isinstance(node.op, ast.Mod):
                result = left % right
            elif isinstance(node.op, ast.Pow):
                if abs(right) > _MAX_POWER_EXPONENT:
                    return _EVAL_ERROR
                result = math.pow(left, right)
            else:
                return _EVAL_ERROR
        except (ArithmeticError, ValueError):
            return _EVAL_ERROR
        normalized = _finite_number(result)
        return normalized if normalized is not None else _EVAL_ERROR
    if isinstance(node, ast.Call):
        if (
            not isinstance(node.func, ast.Name)
            or node.func.id not in _AGGREGATES
            or len(node.args) != 1
            or node.keywords
        ):
            return _EVAL_ERROR
        values = _eval_expression_node(node.args[0], env, next_depth)
        if not isinstance(values, list) or not values or len(values) > _MAX_AGGREGATE_ITEMS:
            return _EVAL_ERROR
        numbers = [_finite_number(value) for value in values]
        if any(number is None for number in numbers):
            return _EVAL_ERROR
        try:
            result = _AGGREGATES[node.func.id](numbers)
        except (ArithmeticError, TypeError, ValueError):
            return _EVAL_ERROR
        normalized = _finite_number(result)
        return normalized if normalized is not None else _EVAL_ERROR
    if isinstance(node, ast.BoolOp):
        values = [_eval_expression_node(item, env, next_depth) for item in node.values]
        if any(not isinstance(value, bool) for value in values):
            return _EVAL_ERROR
        return all(values) if isinstance(node.op, ast.And) else any(values)
    if isinstance(node, ast.Compare):
        left = _eval_expression_node(node.left, env, next_depth)
        if left is _EVAL_ERROR:
            return _EVAL_ERROR
        for op, comparator in zip(node.ops, node.comparators):
            right = _eval_expression_node(comparator, env, next_depth)
            if right is _EVAL_ERROR:
                return _EVAL_ERROR
            try:
                if isinstance(op, ast.Eq):
                    matched = left == right
                elif isinstance(op, ast.NotEq):
                    matched = left != right
                elif isinstance(op, ast.Lt):
                    matched = left < right
                elif isinstance(op, ast.LtE):
                    matched = left <= right
                elif isinstance(op, ast.Gt):
                    matched = left > right
                elif isinstance(op, ast.GtE):
                    matched = left >= right
                elif isinstance(op, ast.In):
                    matched = left in right
                elif isinstance(op, ast.NotIn):
                    matched = left not in right
                elif isinstance(op, ast.Is):
                    matched = left is right
                elif isinstance(op, ast.IsNot):
                    matched = left is not right
                else:
                    return _EVAL_ERROR
            except (TypeError, ValueError):
                return _EVAL_ERROR
            if not matched:
                return False
            left = right
        return True
    return _EVAL_ERROR


def _safe_eval_formula(formula: str, inputs: dict) -> Optional[float]:
    """Evaluate a simple arithmetic expression over numeric inputs.

    Only arithmetic nodes / constants / names are allowed (no function calls,
    no attribute access). The exception is a whitelisted AGGREGATE call
    (sum / avg / mean / median / max / min / count) whose argument is a list
    variable resolved from an array evidence path like ``$.rows[*].field`` —
    so "先对数组求和/均值再参与计算"的推理链也能被代码复核。
    """
    tree = _parse_expression(formula, allow_assignment_label=True)
    if tree is None:
        return None
    allowed = (
        ast.Expression, ast.BinOp, ast.UnaryOp, ast.Constant, ast.Name,
        ast.Add, ast.Sub, ast.Mult, ast.Div, ast.Mod, ast.Pow,
        ast.USub, ast.UAdd, ast.Load, ast.Call, ast.List,
    )
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            func = node.func
            if not (
                isinstance(func, ast.Name)
                and func.id in _AGGREGATES
                and len(node.args) == 1
                and not node.keywords
            ):
                return None  # 只允许白名单聚合函数
        elif not isinstance(node, allowed):
            return None
    env = {k: v for k, v in (inputs or {}).items() if isinstance(v, (int, float, list))}
    result = _eval_expression_node(tree, env)
    return _finite_number(result)


# 白名单聚合函数：数组 evidence（$.rows[*].field）参与表达式的唯一入口
def _agg_sum(seq): return float(sum(seq))

def _agg_avg(seq): return sum(seq) / len(seq)

def _agg_max(seq): return float(max(seq))

def _agg_min(seq): return float(min(seq))

def _agg_count(seq): return float(len(seq))

def _agg_median(seq):
    s = sorted(seq)
    n = len(s)
    if not n:
        return float("nan")
    mid = n // 2
    return float(s[mid]) if n % 2 else (s[mid - 1] + s[mid]) / 2.0


_AGGREGATES = {
    "sum": _agg_sum,
    "avg": _agg_avg,
    "mean": _agg_avg,
    "median": _agg_median,
    "max": _agg_max,
    "min": _agg_min,
    "count": _agg_count,
}


def _fmt(value: float, has_percent: bool = False) -> str:
    if has_percent:
        return f"{value * 100:.2f}%"
    return f"{value:.4f}".rstrip("0").rstrip(".")


def _safe_eval_rule(rule: str, env: dict) -> Optional[bool]:
    """Evaluate a boolean inference rule over resolved evidence values.

    Supports comparisons (>, <, >=, <=, ==, !=, in), boolean operators
    (and / or / not), arithmetic on numeric variables, and string constants
    for equality — all AST-whitelisted, no function calls, no attribute
    access. The LLM never supplies the RESULT: the code evaluates the rule
    from values resolved out of the original tool JSON.
    """
    tree = _parse_expression(rule)
    if tree is None:
        return None
    allowed = (
        ast.Expression, ast.BoolOp, ast.And, ast.Or, ast.Not,
        ast.UnaryOp, ast.USub, ast.UAdd,
        ast.Compare, ast.Eq, ast.NotEq, ast.Lt, ast.LtE, ast.Gt, ast.GtE,
        ast.In, ast.NotIn, ast.Is, ast.IsNot,
        ast.BinOp, ast.Add, ast.Sub, ast.Mult, ast.Div, ast.Mod, ast.Pow,
        ast.Constant, ast.Name, ast.Load, ast.Call, ast.List,
    )
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            func = node.func
            if not (
                isinstance(func, ast.Name)
                and func.id in _AGGREGATES
                and len(node.args) == 1
                and not node.keywords
            ):
                return None  # 只允许白名单聚合函数
        elif not isinstance(node, allowed):
            return None
    result = _eval_expression_node(tree, env)
    return result if isinstance(result, bool) else None


def _bool_conclusion(value: str) -> Optional[bool]:
    """Normalize a claim.value that states a pass/fail or true/false conclusion."""
    text = str(value or "").strip().lower()
    if text in ("通过", "是", "真", "true", "yes", "符合", "满足", "1"):
        return True
    if text in ("未通过", "否", "假", "false", "no", "不符合", "不满足", "0"):
        return False
    return None


def _evidence_dict(entry: Any) -> Optional[dict]:
    if isinstance(entry, dict):
        return dict(entry)
    if hasattr(entry, "model_dump"):
        return entry.model_dump()
    return None


def _index_evidence(evidence: Any) -> tuple[List[dict], Dict[str, dict], List[str]]:
    """Normalize evidence and reject ambiguous aliases before resolution."""
    if not isinstance(evidence, list) or not evidence:
        return [], {}, ["claim.evidence 为空或不是列表"]

    refs: List[dict] = []
    refs_by_alias: Dict[str, dict] = {}
    errors: List[str] = []
    for entry in evidence:
        ref = _evidence_dict(entry)
        if ref is None:
            errors.append("evidence 条目不是有效对象")
            continue
        refs.append(ref)
        alias = str(ref.get("alias") or "").strip()
        if not alias:
            errors.append("evidence.alias 不能为空")
        elif alias in refs_by_alias:
            errors.append(f"evidence.alias {alias!r} 重复，来源存在歧义")
        else:
            refs_by_alias[alias] = ref
    return refs, refs_by_alias, errors


def _numeric_evidence_list(raw_value: list, unit: str, ref: dict) -> tuple[Optional[List[float]], str]:
    """Convert one flat, non-empty, bounded evidence array to finite numbers."""
    location = f"{ref.get('source_tool')}:{ref.get('json_path')}"
    if not raw_value:
        return None, f"{location} 解析到空数组，不能作为计算证据"
    if len(raw_value) > _MAX_AGGREGATE_ITEMS:
        return None, f"{location} 数组超过 {_MAX_AGGREGATE_ITEMS} 项的安全上限"

    numbers: List[float] = []
    for item in raw_value:
        if isinstance(item, (dict, list)):
            return None, f"{location} 解析到嵌套数组/对象；聚合仅支持一维数值数组"
        number = _coerce_evidence_number(item, unit)
        if number is None:
            return None, f"{location} 的元素 {item!r} 不是有限数值"
        numbers.append(number)
    return numbers, ""


def _invalid_claim_item(claim: dict, verification_type: str, refs: List[dict], errors: List[str]) -> FactVerificationItem:
    return FactVerificationItem(
        claim=str(claim.get("claim", "")),
        verification_type=verification_type,
        source_tool=_evidence_tools(refs),
        reported_value=str(claim.get("value", "")),
        expected_value="无法校验",
        failure_reason="；".join(dict.fromkeys(errors)),
        passed=False,
    )


def _check_rule(
    claim: dict,
    payloads: Dict[str, tuple[str, Any]],
) -> FactVerificationItem:
    """推理性/阈值判断：代码从 JSON 路径取值并求值布尔规则，再与结论比对。"""
    claim_text = str(claim.get("claim", ""))
    reported = str(claim.get("value", ""))
    rule = str((claim.get("rule") or "").strip())
    refs, refs_by_alias, errors = _index_evidence(claim.get("evidence"))
    rule_names = _expression_names(rule)
    if rule_names is None:
        errors.append(f"规则 {rule!r} 不是合法或安全的布尔表达式")
    elif not rule_names:
        errors.append("rule 必须引用至少一个 evidence.alias，不能使用常量规则")
    else:
        missing = rule_names - set(refs_by_alias)
        unused = set(refs_by_alias) - rule_names
        if missing:
            errors.append(f"规则变量 {sorted(missing)!r} 缺少同名 evidence")
        if unused:
            errors.append(f"evidence.alias {sorted(unused)!r} 未被 rule 使用")

    env: Dict[str, Any] = {}
    resolved_evidence: List[dict] = []
    for name in sorted(rule_names or set()):
        ref = refs_by_alias.get(name)
        if ref is None:
            continue
        raw_value, error = _resolve_evidence(ref, payloads)
        if raw_value is _MISSING:
            errors.append(error)
            continue
        unit = str(ref.get("unit") or "")
        if isinstance(raw_value, list):
            numbers, list_error = _numeric_evidence_list(raw_value, unit, ref)
            if list_error:
                errors.append(list_error)
            elif numbers is not None:
                env[name] = numbers
        elif isinstance(raw_value, (int, float)) and not isinstance(raw_value, bool):
            number = _coerce_evidence_number(raw_value, unit)
            if number is None:
                errors.append(f"{ref.get('source_tool')}:{ref.get('json_path')} 不是有限数值")
            else:
                env[name] = number
        elif isinstance(raw_value, str) and unit and str(unit).lower() in _PERCENT_UNITS:
            number = _coerce_evidence_number(raw_value, unit)
            if number is None:
                errors.append(f"{ref.get('source_tool')}:{ref.get('json_path')} 不是有效百分比")
            else:
                env[name] = number
        elif raw_value is None or isinstance(raw_value, dict):
            errors.append(
                f"{ref.get('source_tool')}:{ref.get('json_path')} 解析到空值或对象，"
                "rule 只支持标量或一维数值数组"
            )
        else:
            env[name] = raw_value  # 字符串/布尔值原样（支持等值比较）
        resolved_evidence.append(_resolved_evidence(ref, raw_value))

    if errors:
        return FactVerificationItem(
            claim=claim_text,
            verification_type="rule",
            source_tool=_evidence_tools(refs),
            reported_value=reported,
            expected_value="无法求值",
            evidence=resolved_evidence,
            failure_reason="；".join(dict.fromkeys(errors)),
            passed=False,
        )

    result = _safe_eval_rule(rule, env)
    if result is None:
        return FactVerificationItem(
            claim=claim_text,
            verification_type="rule",
            source_tool=_evidence_tools(refs),
            reported_value=reported,
            expected_value="无法求值",
            evidence=resolved_evidence,
            failure_reason=(
                f"规则 {rule!r} 无法在代码中求得布尔值（仅支持比较/布尔/算术表达式）"
            ),
            passed=False,
        )

    conclusion = _bool_conclusion(reported)
    expected_text = "通过" if result else "未通过"
    if conclusion is None:
        return FactVerificationItem(
            claim=claim_text,
            verification_type="rule",
            source_tool=_evidence_tools(refs),
            reported_value=reported,
            expected_value=expected_text,
            evidence=resolved_evidence,
            failure_reason=(
                f"value 必须给出可判定的结论（通过/未通过 或 真/假），得到 {reported!r}"
            ),
            passed=False,
        )

    if conclusion == bool(result):
        return FactVerificationItem(
            claim=claim_text,
            verification_type="rule",
            source_tool=_evidence_tools(refs),
            reported_value=reported,
            expected_value=expected_text,
            evidence=resolved_evidence,
            passed=True,
        )

    return FactVerificationItem(
        claim=claim_text,
        verification_type="rule",
        source_tool=_evidence_tools(refs),
        reported_value=reported,
        expected_value=expected_text,
        evidence=resolved_evidence,
        difference="",
        failure_reason=(
            f"规则 {rule} 由代码求值为 {expected_text}，"
            f"与报告结论 {reported!r} 不符；请修正结论或依据。"
        ),
        passed=False,
    )


def _check_fact(claim: dict, payloads: Dict[str, tuple[str, Any]]) -> FactVerificationItem:
    """Resolve exact evidence fields and compare them with the report value."""
    reported = str(claim.get("value", ""))
    claim_text = str(claim.get("claim", ""))

    refs, _, errors = _index_evidence(claim.get("evidence"))
    if errors:
        return FactVerificationItem(
            claim=claim_text,
            verification_type="fact",
            source_tool=_evidence_tools(refs),
            reported_value=reported,
            expected_value="无法解析",
            passed=False,
            failure_reason="；".join(dict.fromkeys(errors)),
        )

    resolved = []
    resolved_values = []
    errors = []
    for ref in refs:
        value, error = _resolve_evidence(ref, payloads)
        if value is _MISSING:
            errors.append(error)
            continue
        resolved.append(_resolved_evidence(ref, value))
        if isinstance(value, (dict, list)):
            errors.append(
                f"{ref.get('source_tool')}:{ref.get('json_path')} 解析到数组或对象；"
                "普通 fact 必须引用单个标量，数组请使用 formula/rule 聚合"
            )
            continue
        if value is None or (
            isinstance(value, (int, float))
            and not isinstance(value, bool)
            and _coerce_evidence_number(value, str(ref.get("unit") or "")) is None
        ):
            errors.append(f"{ref.get('source_tool')}:{ref.get('json_path')} 解析到空值或非有限数值")
            continue
        resolved_values.append((ref, value))

    source_tools = _evidence_tools(refs)
    if errors:
        return FactVerificationItem(
            claim=claim_text,
            verification_type="fact",
            source_tool=source_tools,
            reported_value=reported,
            expected_value="未找到",
            evidence=resolved,
            passed=False,
            failure_reason="；".join(errors),
        )

    matched = next(
        (
            (ref, value) for ref, value in resolved_values
            if _values_match(reported, value, str(ref.get("unit") or ""))
        ),
        None,
    )
    if matched is not None:
        return FactVerificationItem(
            claim=claim_text,
            verification_type="fact",
            source_tool=source_tools,
            reported_value=reported,
            expected_value=str(matched[1])[:80],
            evidence=resolved,
            passed=True,
        )

    return FactVerificationItem(
        claim=claim_text,
        verification_type="fact",
        source_tool=source_tools,
        reported_value=reported,
        expected_value=" / ".join(item["resolved_value"] for item in resolved)[:120],
        evidence=resolved,
        difference="",
        failure_reason=(
            f"代码从 evidence.json_path 解析出的值与报告值 {reported!r} 不一致；"
            "请修正 value 或证据路径。"
        ),
        passed=False,
    )


def _check_calculation(
    claim: dict,
    payloads: Dict[str, tuple[str, Any]],
) -> FactVerificationItem:
    """Resolve every formula input from raw JSON, then re-evaluate it."""
    claim_text = str(claim.get("claim", ""))
    reported = str(claim.get("value", ""))
    formula = str(claim.get("formula") or "").strip()
    evidence = claim.get("evidence") or []
    reported_num, has_percent = _parse_num(reported)
    formula_names = _formula_names(formula)
    resolved_inputs: Dict[str, Any] = {}
    resolved_evidence: List[dict] = []
    refs, refs_by_alias, errors = _index_evidence(evidence)

    if formula_names is None:
        errors.append(f"公式 {formula!r} 不是合法或安全的算术表达式")
    elif not formula_names:
        errors.append("formula 必须引用至少一个 evidence.alias，不能使用常量公式")
    else:
        missing = formula_names - set(refs_by_alias)
        unused = set(refs_by_alias) - formula_names
        if missing:
            errors.append(f"公式变量 {sorted(missing)!r} 缺少同名 evidence")
        if unused:
            errors.append(f"evidence.alias {sorted(unused)!r} 未被 formula 使用")
        for name in sorted(formula_names):
            ref = refs_by_alias.get(name)
            if ref is None:
                continue
            raw_value, error = _resolve_evidence(ref, payloads)
            if raw_value is _MISSING:
                errors.append(error)
                continue
            unit = str(ref.get("unit") or "")
            if isinstance(raw_value, list):
                numbers, list_error = _numeric_evidence_list(raw_value, unit, ref)
                if list_error:
                    errors.append(list_error)
                elif numbers is not None:
                    resolved_inputs[name] = numbers
            else:
                number = _coerce_evidence_number(raw_value, unit)
                if number is None:
                    errors.append(
                        f"{ref.get('source_tool')}:{ref.get('json_path')} 的值 "
                        f"{raw_value!r} 不是有限数值"
                    )
                    continue
                resolved_inputs[name] = number
            resolved_evidence.append(_resolved_evidence(ref, raw_value))

    computed = None if errors else _safe_eval_formula(formula, resolved_inputs)

    if not formula or computed is None or reported_num is None:
        return FactVerificationItem(
            claim=claim_text,
            verification_type="calculation",
            source_tool=_evidence_tools(refs),
            reported_value=reported,
            expected_value="无法复算",
            evidence=resolved_evidence,
            difference="",
            failure_reason=(
                "计算方式不完整或无法从原始工具数据复算："
                + ("；".join(dict.fromkeys(errors)) if errors else f"formula={formula!r}")
                + "。请为每个公式变量提供精确 evidence.json_path。"
            ),
            passed=False,
        )

    if _close(reported_num, computed):
        return FactVerificationItem(
            claim=claim_text,
            verification_type="calculation",
            source_tool=_evidence_tools(refs),
            reported_value=reported,
            expected_value=_fmt(computed, has_percent),
            evidence=resolved_evidence,
            passed=True,
        )

    return FactVerificationItem(
        claim=claim_text,
        verification_type="calculation",
        source_tool=_evidence_tools(refs),
        reported_value=reported,
        expected_value=_fmt(computed, has_percent),
        evidence=resolved_evidence,
        difference=_fmt(reported_num - computed),
        failure_reason=(
            f"按公式 {formula} 复算得 {_fmt(computed, has_percent)}，"
            f"与报告值 {reported} 不符；请修正计算或数值。"
        ),
        passed=False,
    )


_REPORT_QUANTITY_RE = re.compile(
    r"-?(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?\s*"
    r"(?:个百分点|万亿元|亿元|万元|万股|亿股|%|倍|元|股|万亿|亿|万)"
)


def _coverage_failure(report: str, claims: list) -> Optional[FactVerificationItem]:
    """Conservatively require every unit-bearing report quantity in claims."""
    quantities = list(dict.fromkeys(m.group(0).strip() for m in _REPORT_QUANTITY_RE.finditer(report or "")))
    claim_values = [str((c if isinstance(c, dict) else c.model_dump()).get("value", "")) for c in claims or []]
    uncovered = [
        quantity for quantity in quantities
        if not any(_values_match(quantity, value) and _values_match(value, quantity) for value in claim_values)
    ]
    if not uncovered:
        return None
    shown = "、".join(uncovered[:8])
    if len(uncovered) > 8:
        shown += f" 等 {len(uncovered)} 项"
    return FactVerificationItem(
        claim="报告数值覆盖率",
        verification_type="coverage",
        source_tool="report",
        reported_value=shown,
        expected_value="每个带单位数值均应有对应 claim",
        passed=False,
        failure_reason=f"报告中的数值 {shown} 未出现在 claims.value 中，无法执行代码校验。",
    )


def verify_claims_in_code(
    claims: list,
    raw_data: Dict[str, str],
    report: str = "",
) -> FactVerificationReport:
    """Deterministic code verification of claims against freshly fetched data.

    No LLM involved: fact claims resolve an exact JSON field; calculation
    claims resolve every formula variable from exact JSON paths. Returns a
    report whose feedback lists every failed claim for the retry round.
    """
    payloads = _payload_map(raw_data)

    items: List[FactVerificationItem] = []
    for entry in claims or []:
        claim = _evidence_dict(entry)
        if claim is None:
            items.append(_invalid_claim_item(
                {}, "fact", [], ["claim 不是有效对象"],
            ))
            continue

        formula = str(claim.get("formula") or "").strip()
        rule = str(claim.get("rule") or "").strip()
        refs, _, shape_errors = _index_evidence(claim.get("evidence"))
        if formula and rule:
            shape_errors.append("同一 claim 不能同时填写 formula 和 rule")
        if shape_errors:
            verification_type = "calculation" if formula else "rule" if rule else "fact"
            items.append(_invalid_claim_item(claim, verification_type, refs, shape_errors))
        elif formula:
            items.append(_check_calculation(claim, payloads))
        elif rule:
            items.append(_check_rule(claim, payloads))
        else:
            items.append(_check_fact(claim, payloads))

    coverage = _coverage_failure(report, claims)
    if coverage is not None:
        items.append(coverage)

    failed = [item for item in items if not item.passed]
    overall_passed = not failed
    feedback = ""
    if failed:
        lines = [
            f"以下 {len(failed)} 条声明未通过代码校验，请基于上下文中保留的工具原始数据"
            "修正后重新生成分析（不要再次调用工具）："
        ]
        for item in failed:
            lines.append(
                f"- [{item.verification_type}] {item.claim}：{item.failure_reason}"
            )
        feedback = "\n".join(lines)
    elif not items:
        # 无 claims 时无法执行代码校验（结构化提取失败/不支持）
        feedback = (
            "未能提取结构化声明（claims 为空），无法执行代码校验；"
            "请重新生成报告并为每个数据点给出精确 evidence.json_path 与计算方式。"
        )
        overall_passed = False

    return FactVerificationReport(
        items=items, overall_passed=overall_passed, feedback=feedback
    )


def create_fact_checker(
    analyst_type: str,
    verify_llm: Any = None,  # 保留参数兼容旧调用；代码校验不使用 LLM
    max_rounds: int = 2,
) -> Callable[[Dict[str, Any]], Dict[str, Any]]:
    """Create the fact-checker node for one analyst (code-only verification).

    Args:
        analyst_type: fundamentals / technical / game_theory.
        verify_llm: unused (kept for signature compatibility) — verification
            is deterministic code, no LLM call happens in this node.
        max_rounds: Retry budget before the report is marked unverified.
    """
    def fact_checker_node(state: Dict[str, Any]) -> Dict[str, Any]:
        claim_key = _CLAIM_KEYS[analyst_type]
        report = state.get(_REPORT_KEYS[analyst_type], "") or ""
        claims = state.get(claim_key, []) or []

        vs = (state.get("verification_state") or {}).get(analyst_type) or {}
        attempts = int(vs.get("attempts", 0)) + 1

        # 1) 校验数据 = 分析师**首次请求**保留在分支上下文里的原始返回：
        #    不发起任何新的 HTTP 请求（每个数据接口每次运行只被调用一次，
        #    重试轮同样复用该上下文）。上下文缺失（异常流程）时兜底复拉一次。
        own_tools = VERIFY_TOOLS.get(analyst_type) or []
        own_tool = own_tools[0] if own_tools else None
        raw_data: Dict[str, str] = {}
        if own_tool is not None:
            branch_messages = state.get(f"messages_{analyst_type}", []) or []
            for msg in reversed(branch_messages):
                if isinstance(msg, ToolMessage) and getattr(msg, "name", "") == own_tool.name:
                    raw_data[own_tool.name] = str(msg.content)
                    break
            if not raw_data:
                try:
                    raw_data[own_tool.name] = own_tool.invoke(
                        {
                            "ts_code": state["company_of_interest"],
                            "end_date": state.get("trade_date", ""),
                        }
                    )
                except AnalysisStopped:
                    raise  # 停止信号：不做任何降级，直接中断整个运行
                except Exception as exc:  # noqa: BLE001 - a tool failure must not kill the stage
                    raw_data[own_tool.name] = f"[取数失败] {exc}"

        # 2) 纯代码校验：来源是否属实 + 计算是否正确（无 LLM 调用、无新请求）
        verification = verify_claims_in_code(claims, raw_data, report=report)

        new_vs = {
            "attempts": attempts,
            "passed": bool(verification.overall_passed),
            "feedback": verification.feedback,
            "items": [item.model_dump() for item in verification.items],
            "report_md": render_verification_report(verification),
        }
        # Return ONLY this analyst's key: the channel has a merge reducer so
        # the parallel fact-checkers update it concurrently without clobbering
        # each other (last-write-wins per analyst).
        logger.info(
            "FactChecker-%s: attempt %d/%d %s (%d claims, %d items)",
            analyst_type, attempts, max_rounds,
            "PASSED" if verification.overall_passed else "FAILED",
            len(claims), len(verification.items),
        )
        return {"verification_state": {analyst_type: new_vs}}

    return fact_checker_node


def make_verify_router(
    analyst_type: str,
    max_rounds: int = 2,
) -> Callable[[Dict[str, Any]], str]:
    """Create the conditional router placed after a fact-checker node.

    Returns "retry" when the verification failed and the retry budget is not
    exhausted, "next" otherwise (passed, or budget exhausted -> continue with
    the report marked unverified).
    """

    def router(state: Dict[str, Any]) -> str:
        vs = (state.get("verification_state") or {}).get(analyst_type) or {}
        if vs.get("passed"):
            return "next"
        if int(vs.get("attempts", 0)) >= max_rounds:
            return "next"
        return "retry"

    return router
