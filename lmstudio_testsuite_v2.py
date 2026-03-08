#!/usr/bin/env python3
"""
LM Studio testsuite V2 (normal mode):
- reasoning / coding / tooluse
- tests each model N times (default 3)
- logs JSONL + CSV + compact report.txt

Works with LM Studio OpenAI-compatible server: http://localhost:1234/v1
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
DEFAULT_TEMPERATURE = 0.2
DEFAULT_MAX_TOKENS = 900
DEFAULT_REPEATS = 3
DEFAULT_TIMEOUT_S = 300

# -----------------------------
# Helpers
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
    - handles "1,234.56" and "1.234,56" heuristically
    - handles plain "1234,5" -> 1234.5
    - returns first numeric token
    """
    t = strip_think_blocks(text).strip()
    m = re.search(r"[-+]?\d{1,3}(?:[.,]\d{3})*(?:[.,]\d+)?|[-+]?\d+(?:[.,]\d+)?", t)
    if not m:
        return None
    token = m.group(0)

    if "," in token and "." in token:
        # last separator is decimal
        if token.rfind(",") > token.rfind("."):
            token = token.replace(".", "").replace(",", ".")
        else:
            token = token.replace(",", "")
    else:
        if "," in token and "." not in token:
            token = token.replace(",", ".")
        # only dot: ok

    try:
        return float(token)
    except Exception:
        return None

def extract_two_numbers(text: str) -> Optional[Tuple[float, float]]:
    t = strip_think_blocks(text).strip()
    nums = []
    for m in re.finditer(r"[-+]?\d{1,3}(?:[.,]\d{3})*(?:[.,]\d+)?|[-+]?\d+(?:[.,]\d+)?", t):
        val = extract_first_number(m.group(0))
        if val is not None:
            nums.append(val)
        if len(nums) >= 2:
            return nums[0], nums[1]
    return None

def mean(xs: List[float]) -> float:
    return sum(xs) / len(xs) if xs else float("nan")

def pstdev(xs: List[float]) -> float:
    return statistics.pstdev(xs) if len(xs) >= 2 else 0.0


# -----------------------------
# Validators
# -----------------------------

def exact_number(expected: float, tol: float = 1e-6):
    def _v(text: str) -> Tuple[bool, str]:
        got = extract_first_number(text)
        if got is None:
            return False, "no_number_found"
        diff = abs(got - expected)
        ok = diff <= tol
        return ok, f"got={got} expected={expected} diff={diff} tol={tol}"
    return _v

def must_contain_all(*needles: str):
    needles_n = [normalize(n) for n in needles]
    def _v(text: str) -> Tuple[bool, str]:
        t = normalize(strip_think_blocks(text))
        missing = [n for n in needles_n if n not in t]
        return (len(missing) == 0, f"missing={missing}" if missing else "ok")
    return _v

def two_numbers(expected_a: float, expected_b: float, tol: float = 1e-6):
    """
    Accept output containing at least two numbers; compare the first two numbers in order.
    (Normal mode: not ultra strict; allows extra text.)
    """
    def _v(text: str) -> Tuple[bool, str]:
        got = extract_two_numbers(text)
        if got is None:
            return False, "need_two_numbers"
        a, b = got
        ok = abs(a - expected_a) <= tol and abs(b - expected_b) <= tol
        return ok, f"got=({a},{b}) expected=({expected_a},{expected_b}) tol={tol}"
    return _v


# -----------------------------
# Coding validators
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

def run_python_solution(expected_stdout: str, timeout_s: int = 6):
    expected_n = normalize(expected_stdout)
    def _v(text: str) -> Tuple[bool, str]:
        t = strip_think_blocks(text)
        code = extract_code_block(t) or t
        code = code.strip()
        if not code:
            return False, "empty_code"

        with tempfile.TemporaryDirectory() as td:
            path = os.path.join(td, "solution.py")
            with open(path, "w", encoding="utf-8") as f:
                f.write(code + "\n")

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
                err_msg = str(p.stderr or "").strip()
                err_trunc = err_msg[0:300]
                return False, f"nonzero_exit={p.returncode} stderr={err_trunc}"

            out_msg = str(p.stdout or "")
            out = normalize(out_msg)
            ok = out == expected_n
            return ok, f"stdout={out_msg.strip()} dur={dur:.3f}s expected={expected_stdout}"
    return _v


# -----------------------------
# Tool-use: tools + executor
# -----------------------------

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "calc",
            "description": "Evaluate a simple arithmetic expression using + - * / and parentheses.",
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
]

KV_STORE = {
    "project_codename": "seahorse",
    "release_year": "2026",
    "city": "hamburg",
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

    return "ERROR: unknown tool"


# -----------------------------
# Tests
# -----------------------------

@dataclass
class TestCase:
    id: str
    category: str
    system: str
    user: str
    validator: Any
    uses_tools: bool = False


def build_tests() -> List[TestCase]:
    tests: List[TestCase] = []

    # Reasoning (normal) — numeric
    tests.append(TestCase(
        id="reasoning_01_discount",
        category="reasoning",
        system="Answer with the final answer only.",
        user="A store gives 25% off. The discounted price is 60. What was the original price?",
        validator=exact_number(80.0),
    ))

    tests.append(TestCase(
        id="reasoning_02_split",
        category="reasoning",
        system="Answer with the final answer only.",
        user="Three people split a bill equally. Each pays 14. How much was the total bill?",
        validator=exact_number(42.0),
    ))

    # Reasoning — logical yes/no (soft check)
    tests.append(TestCase(
        id="reasoning_03_syllogism",
        category="reasoning",
        system="Answer with yes or no.",
        user=("Logic: If all bloops are razzies and all razzies are lazzies, "
              "are all bloops definitely lazzies?"),
        validator=must_contain_all("yes"),
    ))

    # Reasoning — 2-number output (semi-hard but still forgiving)
    # x+y=11, x-y=3 => x=7, y=4
    tests.append(TestCase(
        id="reasoning_04_two_numbers",
        category="reasoning",
        system="Return x and y as two numbers in the answer.",
        user="Solve: x + y = 11 and x - y = 3. Return x and y.",
        validator=two_numbers(7.0, 4.0),
    ))

    # Coding — median
    tests.append(TestCase(
        id="coding_01_median",
        category="coding",
        system="Write ONLY Python code. No explanations. Print the answer to stdout.",
        user=("Given nums=[3,1,4,1,5], print the median. "
              "For odd length, median is the middle value after sorting."),
        validator=run_python_solution("3"),
    ))

    # Coding — palindrome
    tests.append(TestCase(
        id="coding_02_palindrome",
        category="coding",
        system="Write ONLY Python code. No explanations. Print the answer to stdout.",
        user=("Implement is_pal(s) that returns True if s is a palindrome ignoring case and non-letters. "
              "Then call it on 'A man, a plan, a canal: Panama' and print the result."),
        validator=run_python_solution("True"),
    ))

    # Tool-use — arithmetic via tool (but pass is based on final number, tools are logged)
    tests.append(TestCase(
        id="tooluse_01_calc",
        category="tooluse",
        system="You may call tools if helpful. If you use a tool, do so via tool calls. Return final answer as a number.",
        user="Compute (17*19) - (23*7). Use the calc tool.",
        validator=exact_number((17*19) - (23*7)),
        uses_tools=True,
    ))

    # Tool-use — kv lookup
    tests.append(TestCase(
        id="tooluse_02_kv",
        category="tooluse",
        system="You may call tools if helpful. If you use a tool, do so via tool calls. Return final answer only.",
        user="What is the value for key 'project_codename'? Use kv_lookup.",
        validator=must_contain_all("seahorse"),
        uses_tools=True,
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
    choices = resp.get("choices", [])
    if not choices:
        return "", []
    choice = choices[0]
    msg = choice.get("message", {})
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
    raw_error = None

    try:
        if test.uses_tools:
            # Allow multi-round tool calling (up to 4 total tool calls)
            resp = chat_completion(
                base_url, api_key, model, messages,
                temperature, max_tokens, timeout_s,
                tools=TOOLS,
                seed=seed
            )
            text, tool_calls = get_text_and_toolcalls(resp)
            tool_calls_seen = len(tool_calls) > 0
            tool_calls_count += len(tool_calls)

            rounds = 0
            while tool_calls and rounds < 4:
                rounds += 1
                messages.append({
                    "role": "assistant",
                    "content": text,
                    "tool_calls": tool_calls,
                })
                for tc in tool_calls:
                    fn = tc["function"]["name"]
                    args = json.loads(tc["function"].get("arguments", "{}") or "{}")
                    result = tool_exec(fn, args)
                    tool_rounds.append({"name": fn, "args": args, "result": result})
                    messages.append({
                        "role": "tool",
                        "tool_call_id": tc["id"],
                        "content": result,
                    })

                resp_next = chat_completion(
                    base_url, api_key, model, messages,
                    temperature, max_tokens, timeout_s,
                    tools=TOOLS,
                    seed=seed
                )
                text, tool_calls = get_text_and_toolcalls(resp_next)
                tool_calls_count += len(tool_calls)

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
        ok, details = test.validator(final_text)

        return {
            "status": "ok",
            "model": model,
            "test_id": test.id,
            "category": test.category,
            "duration_s": round(dur, 4),
            "tool_calls_seen": tool_calls_seen,
            "tool_calls_count": tool_calls_count,
            "tool_rounds": tool_rounds,
            "pass": bool(ok),
            "validator_details": details,
            "output": final_text,
        }

    except Exception as e:
        dur = time.perf_counter() - t0
        raw_error = repr(e)
        return {
            "status": "error",
            "model": model,
            "test_id": test.id,
            "category": test.category,
            "duration_s": round(dur, 4),
            "tool_calls_seen": tool_calls_seen,
            "tool_calls_count": tool_calls_count,
            "tool_rounds": tool_rounds,
            "pass": False,
            "error": raw_error,
            "output": final_text,
        }


def write_report_txt(path: str, summary_rows: List[Dict[str, Any]]):
    by_model: Dict[str, List[Dict[str, Any]]] = {}
    for r in summary_rows:
        by_model.setdefault(r["model"], []).append(r)

    lines: List[str] = []
    lines.append("LM STUDIO TESTSUITE V2 REPORT")
    lines.append(f"Generated: {dt.datetime.now().isoformat(timespec='seconds')}")
    lines.append("")

    for model, rows in sorted(by_model.items()):
        passes = [1 if r["pass"] else 0 for r in rows]
        durs = [float(r["duration_s"]) for r in rows]
        lines.append(f"MODEL: {model}")
        lines.append(f"  Overall pass-rate: {sum(passes)}/{len(passes)} = {sum(passes)/len(passes)*100:.1f}%")
        lines.append(f"  Avg latency: {mean(durs):.3f}s | Std latency: {pstdev(durs):.3f}s")
        lines.append("")

        for cat in sorted(set(r["category"] for r in rows)):
            rr = [r for r in rows if r["category"] == cat]
            p = sum(1 for r in rr if r["pass"])
            d = [float(r["duration_s"]) for r in rr]
            tool_seen_rate = None
            if cat == "tooluse":
                tool_seen_rate = sum(1 for r in rr if r.get("tool_calls_seen")) / len(rr)

            lines.append(f"  Category: {cat}")
            lines.append(f"    pass-rate: {p}/{len(rr)} = {p/len(rr)*100:.1f}%")
            lines.append(f"    avg: {mean(d):.3f}s | std: {pstdev(d):.3f}s")
            if tool_seen_rate is not None:
                lines.append(f"    tool_calls_seen rate: {tool_seen_rate*100:.1f}%")
        lines.append("\n" + ("-" * 60) + "\n")

    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


def compute_config_hash(args: argparse.Namespace) -> str:
    """Berechnet einen (deterministischen) Hash der Argumente, um zu erkennen, ob Resuming erlaubt ist."""
    # Wandle die args in ein sortiertes Dict um
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
    """Lädt den Checkpoint-State. Returniert ein leeres Dict, falls der Hash nicht stimmt, oder das File fehlt."""
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
    ap.add_argument("--seed", type=int, default=None, help="Optional. If server supports it.")
    ap.add_argument("--outdir", default="logs_v2")
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
    state_file = os.path.join(args.outdir, ".lmstudio_test_run_state_v2.json")

    os.makedirs(args.outdir, exist_ok=True)
    stamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # State laden
    run_state = get_checkpoint_state(state_file, config_hash)
    completed = run_state.setdefault("completed_runs", {})
    
    # Append-Mode wenn wir neu starten (bzw. resumieren) 
    resume_mode = len(completed) > 0
    file_mode = "a" if resume_mode else "w"
    
    if resume_mode:
        print(f"Resuming run... ({len(completed)} tests already completed)")
        # Dateinamen aus dem State wiederherstellen, falls vorhanden
        jsonl_path = run_state.get("jsonl_path", os.path.join(args.outdir, f"lmstudio_v2_{stamp}.jsonl"))
        csv_path = run_state.get("csv_path", os.path.join(args.outdir, f"lmstudio_v2_summary_{stamp}.csv"))
        report_path = run_state.get("report_path", os.path.join(args.outdir, f"lmstudio_v2_report_{stamp}.txt"))
    else:
        jsonl_path = os.path.join(args.outdir, f"lmstudio_v2_{stamp}.jsonl")
        csv_path = os.path.join(args.outdir, f"lmstudio_v2_summary_{stamp}.csv")
        report_path = os.path.join(args.outdir, f"lmstudio_v2_report_{stamp}.txt")
        run_state["jsonl_path"] = jsonl_path
        run_state["csv_path"] = csv_path
        run_state["report_path"] = report_path
        save_checkpoint_state(state_file, run_state)

    summary_rows: List[Dict[str, Any]] = []

    # Bestehende Ergebnisse laden, damit sie in CSV/TXT landen (beim Resume)
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
            # Model-spezifische Temperatur holen:
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
                        "duration_s": rec["duration_s"],
                        "status": rec["status"],
                        "tool_calls_seen": rec.get("tool_calls_seen", False),
                        "tool_calls_count": rec.get("tool_calls_count", 0),
                        "validator_details": rec.get("validator_details", ""),
                    })

    with open(csv_path, "w", encoding="utf-8", newline="") as cf:
        fieldnames = [
            "timestamp", "model", "repeat", "test_id", "category",
            "pass", "duration_s", "status",
            "tool_calls_seen", "tool_calls_count",
            "validator_details"
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
