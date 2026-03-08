#!/usr/bin/env python3
"""
LM Studio HARDMODE TestSuite v2
- stricter reasoning / coding / tooluse
- robust number parsing (comma/dot, thousands separators)
- multi-round tool calling (up to N tool rounds)
- logs JSONL + CSV + compact report.txt
- optional --seed (if server supports it)

Works with LM Studio OpenAI-compatible server:
  http://localhost:1234/v1
"""

from __future__ import annotations

import argparse
import ast
import csv
import datetime as dt
import hashlib
import json
import operator
import os
import re
import statistics
import subprocess
import tempfile
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import warnings

from lmstudio_model_utils import (
    DEFAULT_MODELS,
    MODEL_TEMPERATURES,
    format_model_inventory,
    get_model_temperature,
    resolve_models,
)

# Unterdrückt die urllib3 NotOpenSSLWarning für macOS (nur relevant für HTTPS, Script nutzt HTTP)
warnings.filterwarnings("ignore", message="urllib3 v2 only supports OpenSSL 1.1.1+")

import requests


# -----------------------------
# Defaults
# -----------------------------

DEFAULT_BASE_URL = "http://localhost:1234/v1"
DEFAULT_TIMEOUT_S = 180
DEFAULT_REPEATS = 3

DEFAULT_BASE_URL = "http://localhost:1234/v1"
DEFAULT_TIMEOUT_S = 180
DEFAULT_REPEATS = 3

# Hardmode: reduce randomness
DEFAULT_TEMPERATURE = 0.2
DEFAULT_MAX_TOKENS = 1000

# Hardmode tool loop bounds
MAX_TOOL_ROUNDS = 4          # assistant->tools->assistant cycles
MAX_TOOL_CALLS_TOTAL = 8     # total tool calls across rounds


# -----------------------------
# Normalization & number parsing (V2-style robust)
# -----------------------------

def normalize(s: str) -> str:
    return re.sub(r"\s+", " ", s.strip().lower())

def strip_think_blocks(text: str) -> str:
    """Entfernt <think>...</think> oder <thought>...</thought> Blöcke, die von Reasoning-Modellen generiert werden."""
    t = re.sub(r"<(think|thought)>.*?</\1>", "", text, flags=re.DOTALL | re.IGNORECASE)
    # Falls das Modell bei max_tokens abbricht und das Tag nicht schließt:
    t = re.sub(r"<(think|thought)>.*", "", t, flags=re.DOTALL | re.IGNORECASE)
    return t

def extract_first_number(text: str) -> Optional[float]:
    """
    Robust extraction:
    - handles "1,234.56" and "1.234,56" heuristically (last separator = decimal)
    - handles "1234,5" -> 1234.5
    - returns first numeric token
    """
    t = strip_think_blocks(text).strip()
    m = re.search(r"[-+]?\d{1,3}(?:[.,]\d{3})*(?:[.,]\d+)?|[-+]?\d+(?:[.,]\d+)?", t)
    if not m:
        return None
    token = m.group(0)

    if "," in token and "." in token:
        if token.rfind(",") > token.rfind("."):
            token = token.replace(".", "").replace(",", ".")
        else:
            token = token.replace(",", "")
    else:
        if "," in token and "." not in token:
            token = token.replace(",", ".")

    try:
        return float(token)
    except Exception:
        return None

def extract_all_numbers(text: str) -> List[float]:
    nums: List[float] = []
    t = strip_think_blocks(text).strip()
    for m in re.finditer(r"[-+]?\d{1,3}(?:[.,]\d{3})*(?:[.,]\d+)?|[-+]?\d+(?:[.,]\d+)?", t):
        v = extract_first_number(m.group(0))
        if v is not None:
            nums.append(v)
    return nums

def extract_two_numbers(text: str) -> Optional[Tuple[float, float]]:
    nums = extract_all_numbers(text)
    if len(nums) >= 2:
        return nums[0], nums[1]
    return None


# -----------------------------
# "Final answer only" heuristic (Hardmode)
# -----------------------------

def is_final_answer_only(text: str) -> bool:
    """
    Hardmode wants "final answer only" (rough heuristic).
    Allow:
      - a single number (optionally with punctuation)
      - OR yes/no/true/false as single token
      - OR two numbers separated by space (for the 2-number task)
    Disallow:
      - multi-line
      - obvious explanation markers
      - too many words
    """
    t = strip_think_blocks(text).strip()
    if not t:
        return False

    if "\n" in t:
        return False

    nt = normalize(t)
    bad_markers = [
        "because", "therefore", "step", "first", "second", "explain", "reason",
        "i think", "let's", "we", "so that", "hence", "thus"
    ]
    if any(b in nt for b in bad_markers):
        return False

    # yes/no/true/false
    if re.fullmatch(r"(?i)\s*(yes|no|true|false)\s*[\.\!\?]?\s*", t):
        return True

    # Allow "x y" two-number output (still "final")
    parts = t.split()
    if len(parts) == 2 and extract_first_number(parts[0]) is not None and extract_first_number(parts[1]) is not None:
        return True

    # Single-ish number with minimal fluff
    if len(parts) > 6:
        return False

    return extract_first_number(t) is not None


# -----------------------------
# Validators
# -----------------------------

def exact_number(expected: float, tol_abs: float = 1e-6, tol_rel: float = 0.0, require_final_only: bool = True):
    def _v(text: str) -> Tuple[bool, str]:
        if require_final_only and not is_final_answer_only(text):
            return False, "not_final_answer_only"
        got = extract_first_number(text)
        if got is None:
            return False, "no_number_found"
        diff = abs(got - expected)
        ok = diff <= tol_abs or (tol_rel > 0 and diff <= abs(expected) * tol_rel)
        return ok, f"got={got} expected={expected} diff={diff} tol_abs={tol_abs} tol_rel={tol_rel}"
    return _v

def must_equal_token(expected_token: str, require_final_only: bool = True):
    exp = normalize(expected_token)
    def _v(text: str) -> Tuple[bool, str]:
        if require_final_only and not is_final_answer_only(text):
            return False, "not_final_answer_only"
        t = strip_think_blocks(text)
        got = normalize(t).split(" ")[0] if normalize(t) else ""
        ok = got == exp
        return ok, f"got={got} expected={exp}"
    return _v

def two_numbers_ordered(expected_a: float, expected_b: float, tol_abs: float = 1e-6, require_final_only: bool = True):
    def _v(text: str) -> Tuple[bool, str]:
        if require_final_only and not is_final_answer_only(text):
            return False, "not_final_answer_only"
        got = extract_two_numbers(text)
        if got is None:
            return False, "need_two_numbers"
        a, b = got
        ok = abs(a - expected_a) <= tol_abs and abs(b - expected_b) <= tol_abs
        return ok, f"got=({a},{b}) expected=({expected_a},{expected_b}) tol_abs={tol_abs}"
    return _v


# -----------------------------
# Coding: hard validators (unit-test style)
# -----------------------------

def extract_code_block(text: str) -> Optional[str]:
    t = strip_think_blocks(text)
    m = re.search(r"```python\s*(.*?)```", t, flags=re.DOTALL | re.IGNORECASE)
    if m:
        return m.group(1)
    m = re.search(r"```\s*(.*?)```", t, flags=re.DOTALL)
    if m:
        return m.group(1)
    return None

def run_python_with_harness(harness: str, timeout_s: int = 8, require_code_only: bool = True):
    """
    Model must output Python code defining requested functions.
    We append a harness with multiple cases. Harness prints 'OK' on success, else 'FAIL ...'.
    """
    def _v(text: str) -> Tuple[bool, str]:
        code = extract_code_block(text) or text
        code = code.strip()
        if not code:
            return False, "empty_code"

        if require_code_only:
            nt = normalize(strip_think_blocks(text))
            # crude prose detection
            if "here is" in nt or "explanation" in nt or "i will" in nt or "sure" in nt:
                return False, "contains_prose"

        # soft guardrails (not perfect)
        lowered = code.lower()
        forbidden = [
            "import requests", "import urllib", "import socket",
            "subprocess", "os.system", "__import__", "eval(", "exec(",
            # file writes/reads often undesired in benchmark harness
            "open(",
        ]
        if any(f in lowered for f in forbidden):
            return False, "forbidden_token_detected"

        with tempfile.TemporaryDirectory() as td:
            path = os.path.join(td, "solution.py")
            with open(path, "w", encoding="utf-8") as f:
                f.write(code + "\n\n")
                f.write("# --- HARNESS ---\n")
                f.write(harness + "\n")

            try:
                start = time.perf_counter()
                p = subprocess.run(
                    ["python3", "-I", path],
                    capture_output=True,
                    text=True,
                    timeout=timeout_s,
                    cwd=td,
                    env={"PYTHONHASHSEED": "0"},
                )
                dur = time.perf_counter() - start
            except subprocess.TimeoutExpired:
                return False, f"python_timeout>{timeout_s}s"

            if p.returncode != 0:
                return False, f"nonzero_exit={p.returncode} stderr={p.stderr.strip()[:300]} dur={dur:.3f}s"

            out = p.stdout.strip()
            if out != "OK":
                return False, f"unexpected_stdout={out[:300]} dur={dur:.3f}s"

            return True, f"OK dur={dur:.3f}s"

    return _v


# -----------------------------
# Tool-use: tools + executor (multi-round)
# -----------------------------

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "calc",
            "description": "Evaluate an arithmetic expression using + - * / and parentheses.",
            "parameters": {
                "type": "object",
                "properties": {"expression": {"type": "string"}},
                "required": ["expression"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "kv_lookup",
            "description": "Look up a value in a tiny key-value store.",
            "parameters": {
                "type": "object",
                "properties": {"key": {"type": "string"}},
                "required": ["key"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "string_stats",
            "description": "Return stats about a string: length, vowel_count, word_count. Returns JSON.",
            "parameters": {
                "type": "object",
                "properties": {"text": {"type": "string"}},
                "required": ["text"],
            },
        },
    },
]

KV_STORE = {
    "project_codename": "seahorse",
    "release_year": "2026",
    "city": "hamburg",
    "team_size": "7",
}

def _eval_math_ast(node: ast.AST) -> float:
    operators = {
        ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul,
        ast.Div: operator.truediv, ast.USub: operator.neg, ast.UAdd: operator.pos
    }
    if isinstance(node, ast.Constant):
        if not isinstance(node.value, (int, float)):
            raise TypeError("Only numbers allowed")
        return float(node.value)
    elif isinstance(node, ast.BinOp):
        return operators[type(node.op)](_eval_math_ast(node.left), _eval_math_ast(node.right))
    elif isinstance(node, ast.UnaryOp):
        return operators[type(node.op)](_eval_math_ast(node.operand))
    else:
        raise TypeError(f"Unsupported operation: {type(node).__name__}")

def safe_eval_math(expr: str) -> str:
    if not re.fullmatch(r"[0-9\.\+\-\*\/\(\)\s]+", expr or ""):
        return "ERROR: invalid characters"
    try:
        node = ast.parse(expr, mode='eval').body
        val = _eval_math_ast(node)
        return str(val)
    except Exception as e:
        return f"ERROR: {e}"

def tool_exec(name: str, args: Dict[str, Any]) -> str:
    if name == "calc":
        expr = args.get("expression", "")
        return safe_eval_math(expr)

    if name == "kv_lookup":
        key = str(args.get("key", ""))
        return KV_STORE.get(key, "NOT_FOUND")

    if name == "string_stats":
        text = str(args.get("text", ""))
        vowels = sum(1 for c in text.lower() if c in "aeiou")
        words = len([w for w in re.split(r"\s+", text.strip()) if w])
        return json.dumps({"length": len(text), "vowel_count": vowels, "word_count": words})

    return "ERROR: unknown tool"


# -----------------------------
# TestCase
# -----------------------------

@dataclass
class TestCase:
    id: str
    category: str  # reasoning | coding | tooluse
    system: str
    user: str
    validator: Any
    uses_tools: bool = False
    require_tool_calls: bool = False


# -----------------------------
# Hardmode tests (unchanged intent, but uses improved validators)
# -----------------------------

def build_tests() -> List[TestCase]:
    tests: List[TestCase] = []

    # Reasoning hard 1: multi-step numeric
    # 3/8 x = 27 => x=72; 15% of 72 = 10.8
    tests.append(TestCase(
        id="reasoning_hard_01",
        category="reasoning",
        system="Return the final numeric answer only. No explanation.",
        user="If 3/8 of x equals 27, what is 15% of x? Return only the number.",
        validator=exact_number(10.8, tol_abs=1e-6, require_final_only=True),
    ))

    # Reasoning hard 2: two-number output (ordered)
    tests.append(TestCase(
        id="reasoning_hard_02",
        category="reasoning",
        system="Return only two numbers separated by a single space: x y. No other text.",
        user="Solve: x + y = 11 and x - y = 3. Return: x y",
        validator=two_numbers_ordered(7.0, 4.0, tol_abs=1e-6, require_final_only=True),
    ))

    # Reasoning hard 3: strict yes/no
    tests.append(TestCase(
        id="reasoning_hard_03",
        category="reasoning",
        system="Answer with only 'yes' or 'no'.",
        user=("A function is strictly increasing on all real numbers. "
              "Is it possible for it to have two different x values with the same output?"),
        validator=must_equal_token("no", require_final_only=True),
    ))

    # Coding hard 1: parse ints with cases
    harness_1 = r"""
def _run():
    cases = [
        ("a-1 b 2, 003 c+4", [-1, 2, 3, 4]),
        ("no numbers here", []),
        ("-0 +0 10 -20", [0, 0, 10, -20]),
        ("1,2,3", [1,2,3]),
    ]
    for s, exp in cases:
        got = parse_ints(s)
        if got != exp:
            print("FAIL", s, got, exp)
            return
    print("OK")
_run()
"""
    tests.append(TestCase(
        id="coding_hard_01",
        category="coding",
        system="Write ONLY Python code. Define parse_ints(s). No explanation.",
        user=("Implement parse_ints(s): extract all signed integers from string s. "
              "Integers may have optional leading + or -. Ignore everything else. "
              "Return list of ints in appearance order."),
        validator=run_python_with_harness(harness_1, timeout_s=8, require_code_only=True),
    ))

    # Coding hard 2: BFS shortest path
    harness_2 = r"""
def _run():
    grid = [
        "S..#.",
        ".#.#.",
        ".#..E",
        ".####",
        ".....",
    ]
    got = shortest_path_len(grid)
    if got != 10:
        print("FAIL", got, 10)
        return

    grid2 = ["S#E"]
    if shortest_path_len(grid2) is not None:
        print("FAIL", "expected None")
        return

    print("OK")
_run()
"""
    tests.append(TestCase(
        id="coding_hard_02",
        category="coding",
        system="Write ONLY Python code. Define shortest_path_len(grid). Use BFS. Return int length or None. No explanation.",
        user=("Implement shortest_path_len(grid): grid is list of equal-length strings. "
              "'S' start, 'E' end, '#' walls, '.' open. "
              "Return shortest path length (moves) or None if unreachable."),
        validator=run_python_with_harness(harness_2, timeout_s=10, require_code_only=True),
    ))

    # Coding hard 3: merge intervals with sorting + touching intervals
    harness_3 = r"""
def _run():
    cases = [
        ([[1, 3], [2, 6], [8, 10], [15, 18]], [[1, 6], [8, 10], [15, 18]]),
        ([[1, 4], [4, 5]], [[1, 5]]),
        ([[5, 7], [1, 2], [2, 4]], [[1, 4], [5, 7]]),
        ([], []),
    ]
    for intervals, exp in cases:
        got = merge_intervals(intervals)
        if got != exp:
            print("FAIL", intervals, got, exp)
            return
    print("OK")
_run()
"""
    tests.append(TestCase(
        id="coding_hard_03",
        category="coding",
        system="Write ONLY Python code. Define merge_intervals(intervals). No explanation.",
        user=("Implement merge_intervals(intervals): input is a list of [start, end] integer pairs. "
              "Merge overlapping OR touching intervals and return sorted merged intervals."),
        validator=run_python_with_harness(harness_3, timeout_s=10, require_code_only=True),
    ))

    # Tooluse hard 1: enforce tool calls and multi-step chaining
    # kv_lookup team_size -> "7"; calc (7*7)+3 -> 52
    tests.append(TestCase(
        id="tooluse_hard_01",
        category="tooluse",
        system=("You MUST use tools. First get team_size via kv_lookup. Then compute (team_size*team_size)+3 using calc. "
                "Finally return only the final number."),
        user="Do the required tool steps and return the final number only.",
        validator=exact_number(52.0, tol_abs=1e-6, require_final_only=True),
        uses_tools=True,
        require_tool_calls=True,
    ))

    # Tooluse hard 2: tool returns JSON; model must compute sum fields via calc
    # "Hello AI world": length 14, vowels 5, words 3 => 22
    tests.append(TestCase(
        id="tooluse_hard_02",
        category="tooluse",
        system=("You MUST use tools. Call string_stats on the text. Then compute length+vowel_count+word_count using calc. "
                "Return only the final number."),
        user="Text: Hello AI world",
        validator=exact_number(22.0, tol_abs=1e-6, require_final_only=True),
        uses_tools=True,
        require_tool_calls=True,
    ))

    # Tooluse hard 3: chained lookup + arithmetic
    # release_year=2026, team_size=7 -> ((2026-2000)*7)+5 = 187
    tests.append(TestCase(
        id="tooluse_hard_03",
        category="tooluse",
        system=("You MUST use tools. First get release_year and team_size via kv_lookup. "
                "Then compute ((release_year-2000)*team_size)+5 using calc. "
                "Return only the final number."),
        user="Do the required tool steps and return the final number only.",
        validator=exact_number(187.0, tol_abs=1e-6, require_final_only=True),
        uses_tools=True,
        require_tool_calls=True,
    ))

    return tests


# -----------------------------
# OpenAI-compatible client
# -----------------------------

def chat_completion(
    base_url: str,
    api_key: str,
    model: str,
    messages: List[Dict[str, Any]],
    temperature: float,
    max_tokens: int,
    timeout_s: int,
    tools: Optional[List[Dict[str, Any]]] = None,
    seed: Optional[int] = None,
    max_retries: int = 3,
) -> Dict[str, Any]:
    url = f"{base_url}/chat/completions"
    headers = {"Authorization": f"Bearer {api_key}"}
    payload: Dict[str, Any] = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
    }
    if seed is not None:
        payload["seed"] = seed
    if tools is not None:
        payload["tools"] = tools
        payload["tool_choice"] = "auto"

    for attempt in range(1, max_retries + 1):
        try:
            r = requests.post(url, headers=headers, json=payload, timeout=timeout_s)
            r.raise_for_status()
            return r.json()
        except requests.exceptions.RequestException as e:
            if attempt == max_retries:
                raise e
            wait_time = 2 ** attempt
            print(f"  [API Error] {e} -> Retrying in {wait_time}s ({attempt}/{max_retries})...")
            time.sleep(wait_time)
            
    raise RuntimeError("Unreachable")

def get_text_and_toolcalls(resp: Dict[str, Any]) -> Tuple[str, List[Dict[str, Any]]]:
    choice = resp["choices"][0]
    msg = choice["message"]
    content = msg.get("content") or ""
    tool_calls = msg.get("tool_calls") or []
    return content, tool_calls


# -----------------------------
# Runner
# -----------------------------

def warmup_model(base_url: str, api_key: str, model: str, timeout_s: int):
    """Sende einen simplen Dummy-Request, um das Modell initial in den RAM/VRAM zu laden."""
    print(f"\nWarming up model: {model} ...", end="", flush=True)
    try:
        chat_completion(
            base_url, api_key, model,
            [{"role": "user", "content": "Hi"}],
            temperature=0.0, max_tokens=10, timeout_s=timeout_s
        )
        print(" OK")
    except Exception as e:
        print(f" Error during warmup: {e}")

def run_test_for_model(
    base_url: str,
    api_key: str,
    model: str,
    test: TestCase,
    temperature: float,
    max_tokens: int,
    timeout_s: int,
    seed: Optional[int],
) -> Dict[str, Any]:
    messages: List[Dict[str, Any]] = [
        {"role": "system", "content": test.system},
        {"role": "user", "content": test.user},
    ]

    t0 = time.perf_counter()

    tool_rounds: List[Dict[str, Any]] = []
    tool_calls_seen = False
    tool_calls_count = 0
    final_text = ""

    try:
        if test.uses_tools:
            # Multi-round tool calling until no tool_calls or limits reached.
            resp = chat_completion(
                base_url, api_key, model, messages,
                temperature, max_tokens, timeout_s,
                tools=TOOLS,
                seed=seed
            )
            text, tool_calls = get_text_and_toolcalls(resp)

            rounds = 0
            while tool_calls and rounds < MAX_TOOL_ROUNDS and tool_calls_count < MAX_TOOL_CALLS_TOTAL:
                rounds += 1
                tool_calls_seen = tool_calls_seen or (len(tool_calls) > 0)
                tool_calls_count += len(tool_calls)

                # Append assistant message WITH tool calls
                messages.append({
                    "role": "assistant",
                    "content": text,
                    "tool_calls": tool_calls,
                })

                # Execute each tool call, append tool results
                for tc in tool_calls:
                    if tool_calls_count > MAX_TOOL_CALLS_TOTAL:
                        break
                    fn = tc["function"]["name"]
                    args = json.loads(tc["function"].get("arguments", "{}") or "{}")
                    result = tool_exec(fn, args)
                    tool_rounds.append({"name": fn, "args": args, "result": result})
                    messages.append({
                        "role": "tool",
                        "tool_call_id": tc["id"],
                        "content": result,
                    })

                # Ask again, still with tools enabled
                resp_next = chat_completion(
                    base_url, api_key, model, messages,
                    temperature, max_tokens, timeout_s,
                    tools=TOOLS,
                    seed=seed
                )
                text, tool_calls = get_text_and_toolcalls(resp_next)

            final_text = (text or "").strip()

        else:
            resp = chat_completion(
                base_url, api_key, model, messages,
                temperature, max_tokens, timeout_s,
                seed=seed
            )
            text, _ = get_text_and_toolcalls(resp)
            final_text = (text or "").strip()

        dur = time.perf_counter() - t0

        # Tool compliance (hard requirement for tool tests)
        tool_compliance_ok = True
        tool_compliance_detail = "n/a"
        if test.uses_tools and test.require_tool_calls:
            tool_compliance_ok = tool_calls_seen and tool_calls_count >= 1
            tool_compliance_detail = f"tool_calls_seen={tool_calls_seen} tool_calls_count={tool_calls_count}"

        answer_ok, details = test.validator(final_text)
        pass_final = bool(answer_ok) and bool(tool_compliance_ok)

        return {
            "status": "ok",
            "model": model,
            "test_id": test.id,
            "category": test.category,
            "duration_s": round(dur, 4),
            "pass": pass_final,
            "answer_pass": bool(answer_ok),
            "tool_compliance_pass": bool(tool_compliance_ok),
            "tool_compliance_detail": tool_compliance_detail,
            "validator_details": details,
            "tool_calls_seen": tool_calls_seen,
            "tool_calls_count": tool_calls_count,
            "tool_rounds": tool_rounds,
            "output": final_text,
        }

    except Exception as e:
        dur = time.perf_counter() - t0
        return {
            "status": "error",
            "model": model,
            "test_id": test.id,
            "category": test.category,
            "duration_s": round(dur, 4),
            "pass": False,
            "answer_pass": False,
            "tool_compliance_pass": False,
            "error": repr(e),
            "tool_calls_seen": tool_calls_seen,
            "tool_calls_count": tool_calls_count,
            "tool_rounds": tool_rounds,
            "output": final_text,
        }


# -----------------------------
# Reporting (V2-style compact report)
# -----------------------------

def mean(xs: List[float]) -> float:
    return sum(xs) / len(xs) if xs else float("nan")

def pstdev(xs: List[float]) -> float:
    return statistics.pstdev(xs) if len(xs) >= 2 else 0.0

def write_report_txt(path: str, rows: List[Dict[str, Any]]):
    by_model: Dict[str, List[Dict[str, Any]]] = {}
    for r in rows:
        by_model.setdefault(r["model"], []).append(r)

    lines: List[str] = []
    lines.append("LM STUDIO HARDMODE V2 REPORT")
    lines.append(f"Generated: {dt.datetime.now().isoformat(timespec='seconds')}")
    lines.append("")

    for model, mr in sorted(by_model.items()):
        passes = [1 if r["pass"] else 0 for r in mr]
        durs = [float(r["duration_s"]) for r in mr]
        lines.append(f"MODEL: {model}")
        lines.append(f"  Overall pass-rate: {sum(passes)}/{len(passes)} = {sum(passes)/len(passes)*100:.1f}%")
        lines.append(f"  Avg latency: {mean(durs):.3f}s | Std latency: {pstdev(durs):.3f}s")

        # Tool compliance & answer pass rates (overall)
        ans_passes = [1 if r.get("answer_pass") else 0 for r in mr]
        tool_passes = [1 if r.get("tool_compliance_pass") else 0 for r in mr]
        # only meaningful if there are tooluse tests; still print for visibility
        lines.append(f"  Answer pass-rate: {sum(ans_passes)}/{len(ans_passes)} = {sum(ans_passes)/len(ans_passes)*100:.1f}%")
        lines.append(f"  Tool compliance pass-rate: {sum(tool_passes)}/{len(tool_passes)} = {sum(tool_passes)/len(tool_passes)*100:.1f}%")
        lines.append("")

        # per category breakdown
        for cat in sorted(set(r["category"] for r in mr)):
            rr = [r for r in mr if r["category"] == cat]
            p = sum(1 for r in rr if r["pass"])
            d = [float(r["duration_s"]) for r in rr]
            lines.append(f"  Category: {cat}")
            lines.append(f"    pass-rate: {p}/{len(rr)} = {p/len(rr)*100:.1f}%")
            lines.append(f"    avg: {mean(d):.3f}s | std: {pstdev(d):.3f}s")
            if cat == "tooluse":
                tool_seen = sum(1 for r in rr if r.get("tool_calls_seen"))
                lines.append(f"    tool_calls_seen rate: {tool_seen}/{len(rr)} = {tool_seen/len(rr)*100:.1f}%")
        lines.append("\n" + ("-" * 60) + "\n")

    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


# -----------------------------
# Main
# -----------------------------

def compute_config_hash(args: argparse.Namespace) -> str:
    """Berechnet einen (deterministischen) Hash der Argumente, um zu erkennen, ob Resuming erlaubt ist."""
    config_dict = {
        "models": sorted(args.models),
        "model_source": args.model_source,
        "repeats": args.repeats,
        "temperature": args.temperature,
        "max_tokens": args.max_tokens,
        "seed": args.seed,
        "model_temperatures": MODEL_TEMPERATURES,
    }
    config_str = json.dumps(config_dict, sort_keys=True)
    return hashlib.md5(config_str.encode("utf-8")).hexdigest()

def build_run_id(model: str, repeat: int, test_id: str, temperature: float) -> str:
    """Builds a deterministic run id that treats different temperatures as separate runs."""
    temp_key = f"{temperature:.8g}"
    return f"{model}___temp{temp_key}___rep{repeat}___{test_id}"

def get_checkpoint_state(state_file: str, current_hash: str) -> Dict[str, Any]:
    if os.path.exists(state_file):
        try:
            with open(state_file, "r", encoding="utf-8") as f:
                state = json.load(f)
            if state.get("config_hash") == current_hash:
                return state
            else:
                print(f"Config geändert (neuer Hash {current_hash}). Starte von vorne.")
        except Exception as e:
            print(f"Fehler beim Laden des States: {e}. Starte von vorne.")
    return {"config_hash": current_hash, "completed_runs": {}}

def save_checkpoint_state(state_file: str, state: Dict[str, Any]):
    with open(state_file, "w", encoding="utf-8") as f:
        json.dump(state, f, indent=2)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-url", default=DEFAULT_BASE_URL)
    ap.add_argument("--api-key", default="lm-studio")
    ap.add_argument("--models", nargs="*", default=None)
    ap.add_argument("--model-source", choices=["auto", "api", "defaults"], default="auto")
    ap.add_argument("--models-root", default=os.path.expanduser("~/.lmstudio/models"))
    ap.add_argument("--list-available-models", action="store_true")
    ap.add_argument("--repeats", type=int, default=DEFAULT_REPEATS)
    ap.add_argument("--temperature", type=float, default=DEFAULT_TEMPERATURE)
    ap.add_argument("--max-tokens", type=int, default=DEFAULT_MAX_TOKENS)
    ap.add_argument("--timeout", type=int, default=DEFAULT_TIMEOUT_S)
    ap.add_argument("--seed", type=int, default=None, help="Optional, if server supports it.")
    ap.add_argument("--outdir", default="logs_hard_v2")
    args = ap.parse_args()

    selected_models, model_info = resolve_models(
        explicit_models=args.models,
        default_models=DEFAULT_MODELS,
        model_source=args.model_source,
        base_url=args.base_url,
        api_key=args.api_key,
        timeout_s=min(args.timeout, 15),
        models_root=args.models_root,
    )
    args.models = selected_models

    if args.list_available_models:
        print(format_model_inventory(model_info))
        return

    if not args.models:
        raise SystemExit("No LLM models selected. Check /v1/models or pass --models explicitly.")

    print(f"Model source: {model_info['selected_source']} -> {len(args.models)} model(s)")
    if model_info.get("api_models"):
        extra_models = [model for model in model_info["api_models"] if model not in DEFAULT_MODELS]
        if extra_models:
            print("Additional live models discovered:")
            for model in extra_models:
                print(f"  - {model}")
    elif model_info.get("api_error") and args.model_source == "auto":
        print(f"Live model lookup unavailable, using default list: {model_info['api_error']}")

    unavailable = model_info.get("unavailable_explicit_models", [])
    if unavailable:
        print("Warning: these explicit models are not currently reported by /v1/models:")
        for model in unavailable:
            print(f"  - {model}")

    tests = build_tests()
    config_hash = compute_config_hash(args)
    state_file = os.path.join(args.outdir, ".lmstudio_test_run_state_hard_v2.json")

    os.makedirs(args.outdir, exist_ok=True)
    stamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    
    run_state = get_checkpoint_state(state_file, config_hash)
    completed = run_state.setdefault("completed_runs", {})
    
    resume_mode = len(completed) > 0
    file_mode = "a" if resume_mode else "w"
    
    if resume_mode:
        print(f"Resuming run... ({len(completed)} tests already completed)")
        jsonl_path = run_state.get("jsonl_path", os.path.join(args.outdir, f"lmstudio_hard_v2_{stamp}.jsonl"))
        csv_path = run_state.get("csv_path", os.path.join(args.outdir, f"lmstudio_hard_v2_summary_{stamp}.csv"))
        report_path = run_state.get("report_path", os.path.join(args.outdir, f"lmstudio_hard_v2_report_{stamp}.txt"))
    else:
        jsonl_path = os.path.join(args.outdir, f"lmstudio_hard_v2_{stamp}.jsonl")
        csv_path = os.path.join(args.outdir, f"lmstudio_hard_v2_summary_{stamp}.csv")
        report_path = os.path.join(args.outdir, f"lmstudio_hard_v2_report_{stamp}.txt")
        run_state["jsonl_path"] = jsonl_path
        run_state["csv_path"] = csv_path
        run_state["report_path"] = report_path
        save_checkpoint_state(state_file, run_state)

    summary_rows: List[Dict[str, Any]] = []

    if resume_mode and os.path.exists(jsonl_path):
        with open(jsonl_path, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    try:
                        rec = json.loads(line)
                        summary_rows.append(rec)
                    except Exception:
                        pass

    with open(jsonl_path, file_mode, encoding="utf-8") as jf:
        for model in args.models:
            model_temp = get_model_temperature(model, args.temperature)
            
            warmup_model(args.base_url, args.api_key, model, args.timeout)
            for rep in range(1, args.repeats + 1):
                print(f"\n--- Iteration {rep}/{args.repeats} for model: {model} (Temp: {model_temp}) ---")
                for test in tests:
                    run_id = build_run_id(model=model, repeat=rep, test_id=test.id, temperature=model_temp)
                    if run_id in completed:
                        print(f"  [SKIP] {test.id:<25} (already done @ temp={model_temp:.8g})")
                        continue

                    rec = run_test_for_model(
                        base_url=args.base_url,
                        api_key=args.api_key,
                        model=model,
                        test=test,
                        temperature=model_temp,
                        max_tokens=args.max_tokens,
                        timeout_s=args.timeout,
                        seed=args.seed,
                    )
                    
                    status_str = "PASS" if rec["pass"] else "FAIL"
                    if rec.get("status") == "error":
                        status_str = "ERR "
                    tool_str = " (Tools used)" if rec.get("tool_calls_seen") else ""
                    print(f"  [{status_str}] {test.id:<25} ({rec['duration_s']:>6.2f}s){tool_str}")
                    
                    rec["repeat"] = rep
                    rec["timestamp"] = dt.datetime.now().isoformat(timespec="seconds")

                    jf.write(json.dumps(rec, ensure_ascii=False) + "\n")
                    jf.flush()
                    
                    completed[run_id] = True
                    save_checkpoint_state(state_file, run_state)

                    summary_rows.append({
                        "timestamp": rec["timestamp"],
                        "model": rec["model"],
                        "repeat": rec["repeat"],
                        "test_id": rec["test_id"],
                        "category": rec["category"],
                        "pass": rec["pass"],
                        "answer_pass": rec.get("answer_pass", False),
                        "tool_compliance_pass": rec.get("tool_compliance_pass", False),
                        "duration_s": rec["duration_s"],
                        "status": rec["status"],
                        "tool_calls_seen": rec.get("tool_calls_seen", False),
                        "tool_calls_count": rec.get("tool_calls_count", 0),
                        "validator_details": rec.get("validator_details", ""),
                        "tool_compliance_detail": rec.get("tool_compliance_detail", ""),
                    })

    with open(csv_path, "w", encoding="utf-8", newline="") as cf:
        fieldnames = [
            "timestamp", "model", "repeat", "test_id", "category",
            "pass", "answer_pass", "tool_compliance_pass",
            "duration_s", "status",
            "tool_calls_seen", "tool_calls_count",
            "validator_details", "tool_compliance_detail",
        ]
        w = csv.DictWriter(cf, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(summary_rows)

    write_report_txt(report_path, summary_rows)

    print(f"Wrote JSONL log: {jsonl_path}")
    print(f"Wrote CSV summary: {csv_path}")
    print(f"Wrote TXT report: {report_path}")


if __name__ == "__main__":
    main()
