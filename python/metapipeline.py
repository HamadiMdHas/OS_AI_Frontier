#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Oct 31 20:05:49 2026

@author: hamadih
"""

import argparse
import json
import os
import re
import sys
import textwrap
from pathlib import Path

import requests


OUTPUT_DIR = Path("outputs")

DEFAULT_INPUT = OUTPUT_DIR / "input.txt"
DEFAULT_JSON = OUTPUT_DIR / "out.json"
DEFAULT_REPORT = OUTPUT_DIR / "output.txt"

VLLM_URL = os.environ.get("LLM_URL", "")
MODEL = os.environ.get("LLM_MODEL", "meta-llama/Llama-3.1-70B-Instruct")
API_KEY = os.environ.get("LLM_API_KEY", "")

MAX_TOKENS = 1024
REQUEST_TIMEOUT = 600
TEMPERATURE = 0.2
MAX_RETRIES = 1

MAX_INPUT_CHARS = 8000


STUDY_CONTEXT = """
Study: Structural Topic Model (STM) analysis of DeepSeek user reviews
posted on Google Play.

Topic prevalence model:
    prevalence ~ score + period + score:period

Period coding:
    period = 0: earlier period
    period = 1: later period

Rules:
- Distinguish score, period, and score:period.
- Do not assume what score measures unless the input specifies it.
- Interpret coefficient directions, uncertainty, significance, and
  effect sizes only when supported by the supplied results.
- Do not invent findings, statistics, topic labels, or review quotations.
- Topic words alone do not establish user intentions or experiences.
- Distinguish observations, interpretations, and hypotheses.
- Association does not establish causation.
- Ip must remain descriptive and must make no causal claims.
- State when important information is missing.
"""


def string_schema():
    return {"type": "string"}


def integer_schema():
    return {"type": "integer"}


def array_schema(item):
    return {"type": "array", "items": item}


def object_schema(properties):
    return {
        "type": "object",
        "properties": properties,
        "required": list(properties),
    }


IP_SCHEMA = object_schema({
    "patterns": array_schema(object_schema({
        "id": integer_schema(),
        "description": string_schema(),
    })),
    "interpretations": array_schema(object_schema({
        "id": integer_schema(),
        "text": string_schema(),
        "based_on_patterns": array_schema(integer_schema()),
    })),
    "knowledge_bodies": array_schema(object_schema({
        "name": string_schema(),
        "why_relevant": string_schema(),
        "interpretations": array_schema(integer_schema()),
    })),
})

IF_SCHEMA = object_schema({
    "critiques": array_schema(object_schema({
        "interpretation_id": integer_schema(),
        "critique": string_schema(),
    })),
    "inferences": array_schema(object_schema({
        "id": integer_schema(),
        "hypothesis": string_schema(),
        "supported_by_interpretations": array_schema(integer_schema()),
        "plausibility": {
            "type": "string",
            "enum": ["low", "medium", "high"],
        },
        "rival_explanation": string_schema(),
        "how_to_falsify": string_schema(),
    })),
})

MA_SCHEMA = object_schema({
    "assessment": object_schema({
        "interpretations": string_schema(),
        "inferences": string_schema(),
        "knowledge_bodies": string_schema(),
    }),
    "risks": array_schema(string_schema()),
    "next_steps": array_schema(object_schema({
        "kind": {
            "type": "string",
            "enum": ["analytical", "semantic_intervention"],
        },
        "step": string_schema(),
        "rationale": string_schema(),
    })),
})


TAG_RULES = """
Use numbered references such as [pattern 1], [interpretation 1],
and [inference 1] when relevant. Refer only to IDs that exist.
Return one valid JSON object matching the requested schema.
Keep responses concise. Do not include Markdown fences around JSON.
"""

IP_SYSTEM = """
You are Ip, the descriptive INTERPRETER.

Analyze the supplied STM results, topic words, coefficients, and other
research material.

Produce:
1. Salient observable patterns, numbered from 1.
2. Cautious interpretations supported by those patterns, numbered from 1.
3. Relevant bodies of knowledge or research literatures.

Do not make causal claims. Do not infer statistical significance or coefficient
direction without evidence in the input. Do not assume what score measures.
Do not invent results. Distinguish topic-word semantics from actual user
intentions or experiences. Acknowledge missing information.

""" + TAG_RULES

IF_SYSTEM = """
You are If, the critical INFERENCER.

Review the original evidence and Ip's output.

Produce:
1. Critiques of Ip's interpretations.
2. Plausible, falsifiable hypotheses.
3. A rival explanation for each hypothesis.
4. A concrete way to test or falsify each hypothesis.
5. A low, medium, or high plausibility rating.

Clearly distinguish hypotheses from established findings. Identify alternative
explanations, uncertainty, and additional data needed. Do not invent
coefficients or claim unsupported statistical significance. Distinguish score,
period, and score:period. Cite existing pattern and interpretation IDs.

""" + TAG_RULES

MA_SYSTEM = """
You are Ma, the META-ANALYSIS agent.

Audit the outputs from Ip and If. Identify:
- Unsupported interpretations or hypotheses.
- Overconfident semantic claims.
- Statistical or methodological risks.
- Missing uncertainty checks and alternative explanations.
- Whether proposed hypotheses are testable.

Recommend concise, actionable next steps in two categories.

ANALYTICAL:
Robustness checks, model specification checks, topic-count sensitivity,
uncertainty estimates, coefficient interpretation, metadata validation,
and inspection of representative reviews.

SEMANTIC_INTERVENTION:
Lemmatization, stopword changes, synonym merging, phrase detection,
topic splitting or pooling, keyword-in-context inspection, entity masking,
and rerunning the model after removing dominant terms.

Choose steps relevant to the supplied results. Do not invent findings.
Keep the output concise and cite existing IDs where appropriate.

""" + TAG_RULES


def compact_json(data):
    return json.dumps(
        data,
        ensure_ascii=False,
        separators=(",", ":"),
    )


def extract_tags(text):
    return {
        (kind.lower(), int(number))
        for kind, number in re.findall(
            r"\[(pattern|interpretation|inference)\s+(\d+)\]",
            text,
            flags=re.IGNORECASE,
        )
    }


def all_strings(value):
    if isinstance(value, str):
        yield value
    elif isinstance(value, dict):
        for item in value.values():
            yield from all_strings(item)
    elif isinstance(value, list):
        for item in value:
            yield from all_strings(item)


def save_json(path, data):
    path = Path(path).expanduser()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(data, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def save_text(path, text):
    path = Path(path).expanduser()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text.rstrip() + "\n", encoding="utf-8")


def call_llm(system, user, schema):
    payload = {
        "model": MODEL,
        "messages": [
            {
                "role": "system",
                "content": STUDY_CONTEXT + "\n" + system,
            },
            {
                "role": "user",
                "content": user,
            },
        ],
        "temperature": TEMPERATURE,
        "max_tokens": MAX_TOKENS,
        "stream": False,
        "guided_json": schema,
    }

    headers = {"Content-Type": "application/json"}
    if API_KEY:
        headers["Authorization"] = f"Bearer {API_KEY}"

    try:
        response = requests.post(
            VLLM_URL,
            headers=headers,
            json=payload,
            timeout=REQUEST_TIMEOUT,
        )
    except requests.exceptions.ConnectionError as exc:
        raise RuntimeError(
            "Could not connect to the model server at "
            f"{VLLM_URL}.\n"
            "Check that the server is running and reachable."
        ) from exc
    except requests.exceptions.Timeout as exc:
        raise RuntimeError(
            f"The model request timed out after {REQUEST_TIMEOUT} seconds."
        ) from exc

    if not response.ok:
        try:
            detail = response.json().get("message", response.text)
        except ValueError:
            detail = response.text

        raise RuntimeError(
            f"Server returned HTTP {response.status_code}:\n{detail}\n\n"
            f"Request completion budget: {MAX_TOKENS} tokens. "
            "If this is a context-length error, reduce the prompt size "
            "or set MAX_TOKENS to 512."
        )

    try:
        content = response.json()["choices"][0]["message"]["content"]
        result = json.loads(content)
    except (ValueError, KeyError, IndexError, TypeError) as exc:
        raise RuntimeError(
            "Could not parse the model's JSON response:\n"
            + response.text[:2000]
        ) from exc

    if not isinstance(result, dict):
        raise RuntimeError("The model response is not a JSON object.")

    return result


def validate_ip(data):
    problems = []

    patterns = data.get("patterns", [])
    interpretations = data.get("interpretations", [])

    pids = {p["id"] for p in patterns}
    iids = {i["id"] for i in interpretations}

    if not patterns:
        problems.append("Ip returned no patterns.")
    if not interpretations:
        problems.append("Ip returned no interpretations.")

    for interpretation in interpretations:
        for pid in interpretation.get("based_on_patterns", []):
            if pid not in pids:
                problems.append(
                    f"Interpretation {interpretation['id']} cites "
                    f"nonexistent pattern {pid}."
                )

    for text in all_strings(data):
        for kind, number in extract_tags(text):
            if kind == "pattern" and number not in pids:
                problems.append(f"Invalid [pattern {number}] tag.")
            if kind == "interpretation" and number not in iids:
                problems.append(
                    f"Invalid [interpretation {number}] tag."
                )
            if kind == "inference":
                problems.append("Ip should not cite inference tags.")

    causal_terms = re.compile(
        r"\b(causes?|caused|causing|leads to|led to|"
        r"results in|driven by|drives|is responsible for)\b",
        re.IGNORECASE,
    )
    for text in all_strings(data):
        if causal_terms.search(text):
            problems.append(
                "Potential causal wording in Ip output: "
                + text[:100]
            )

    return problems


def validate_if(data, ip):
    problems = []

    pids = {p["id"] for p in ip["patterns"]}
    iids = {i["id"] for i in ip["interpretations"]}
    fids = {f["id"] for f in data.get("inferences", [])}

    if not fids:
        problems.append("If returned no hypotheses.")

    for critique in data.get("critiques", []):
        iid = critique["interpretation_id"]
        if iid not in iids:
            problems.append(
                f"Critique references nonexistent interpretation {iid}."
            )

    for inference in data.get("inferences", []):
        for iid in inference.get("supported_by_interpretations", []):
            if iid not in iids:
                problems.append(
                    f"Inference {inference['id']} references "
                    f"nonexistent interpretation {iid}."
                )

    for text in all_strings(data):
        for kind, number in extract_tags(text):
            available = {
                "pattern": pids,
                "interpretation": iids,
                "inference": fids,
            }.get(kind, set())
            if number not in available:
                problems.append(f"Invalid [{kind} {number}] tag.")

    return problems


def validate_ma(data, ip, inf):
    problems = []

    if not any(
        step.get("kind") == "semantic_intervention"
        for step in data.get("next_steps", [])
    ):
        problems.append(
            "Ma must include a semantic_intervention next step."
        )

    valid = {
        "pattern": {p["id"] for p in ip["patterns"]},
        "interpretation": {
            i["id"] for i in ip["interpretations"]
        },
        "inference": {
            i["id"] for i in inf["inferences"]
        },
    }

    for text in all_strings(data):
        for kind, number in extract_tags(text):
            if number not in valid.get(kind, set()):
                problems.append(f"Invalid [{kind} {number}] tag.")

    return problems


def run_agent(name, system, prompt, schema, validator, context):
    last_data = None
    last_problems = []

    for attempt in range(MAX_RETRIES + 1):
        current_prompt = prompt

        if attempt:
            feedback = "; ".join(last_problems[:4])
            current_prompt += (
                "\n\nPlease correct these validation problems: "
                + feedback
                + ". Return the complete corrected JSON. Keep it concise."
            )

        print(f"[{name}] sending request...")
        last_data = call_llm(system, current_prompt, schema)

        try:
            last_problems = validator(last_data, *context)
        except Exception as exc:
            last_problems = [f"Validation error: {exc}"]

        if not last_problems:
            print(f"[{name}] validation passed.")
            return last_data, []

        print(
            f"[{name}] validation warnings: {last_problems[:3]}",
            file=sys.stderr,
        )

    return last_data or {}, last_problems


def wrap(text, indent=2, width=100):
    return textwrap.fill(
        str(text),
        width=width,
        initial_indent=" " * indent,
        subsequent_indent=" " * (indent + 2),
    )


def render_report(result):
    lines = [
        "=" * 90,
        "THREE-AGENT STM ANALYSIS",
        f"Model: {result['model']}",
        "Data: DeepSeek user reviews on Google Play",
        "Model: prevalence ~ score + period + score:period",
        "Period: 0 = earlier; 1 = later",
        "=" * 90,
        "",
        "IP — DESCRIPTIVE INTERPRETER",
        "-" * 90,
        "",
        "PATTERNS",
    ]

    ip = result.get("Ip", {})
    inf = result.get("If", {})
    ma = result.get("Ma", {})

    for item in ip.get("patterns", []):
        lines.append(
            wrap(f"[pattern {item['id']}] {item['description']}")
        )

    lines.extend(["", "INTERPRETATIONS"])

    for item in ip.get("interpretations", []):
        lines.append(
            wrap(f"[interpretation {item['id']}] {item['text']}")
        )

    lines.extend(["", "BODIES OF KNOWLEDGE"])

    for item in ip.get("knowledge_bodies", []):
        lines.append(
            wrap(f"{item['name']}: {item['why_relevant']}")
        )

    lines.extend([
        "",
        "IF — CRITICAL INFERENCER",
        "-" * 90,
        "",
        "CRITIQUES",
    ])

    for item in inf.get("critiques", []):
        lines.append(
            wrap(
                f"[interpretation {item['interpretation_id']}] "
                f"{item['critique']}"
            )
        )

    lines.extend(["", "HYPOTHESES"])

    for item in inf.get("inferences", []):
        lines.append(
            wrap(
                f"[inference {item['id']}] {item['hypothesis']} "
                f"(plausibility: {item['plausibility']})"
            )
        )
        lines.append(wrap("Rival explanation: " + item["rival_explanation"], 4))
        lines.append(wrap("How to falsify: " + item["how_to_falsify"], 4))

    lines.extend([
        "",
        "MA — META AUDIT",
        "-" * 90,
    ])

    assessment = ma.get("assessment", {})
    for key, heading in [
        ("interpretations", "Interpretation assessment"),
        ("inferences", "Inference assessment"),
        ("knowledge_bodies", "Knowledge-body assessment"),
    ]:
        if assessment.get(key):
            lines.append(wrap(f"{heading}: {assessment[key]}"))

    lines.extend(["", "RISKS"])

    for risk in ma.get("risks", []):
        lines.append(wrap("- " + risk))

    for kind, heading in [
        ("analytical", "ANALYTICAL NEXT STEPS"),
        ("semantic_intervention", "SEMANTIC INTERVENTIONS"),
    ]:
        steps = [
            s for s in ma.get("next_steps", [])
            if s.get("kind") == kind
        ]
        if steps:
            lines.extend(["", heading])
            for step in steps:
                lines.append(
                    wrap(f"- {step['step']} — {step['rationale']}")
                )

    warnings = result.get("validation_warnings", {})
    if any(warnings.values()):
        lines.extend(["", "VALIDATION WARNINGS"])
        lines.append(json.dumps(warnings, indent=2, ensure_ascii=False))

    if result.get("pipeline_error"):
        lines.extend([
            "",
            "PIPELINE ERROR",
            str(result["pipeline_error"]),
        ])

    return "\n".join(lines)


def run_pipeline(human_input):
    result = {
        "model": MODEL,
        "study_context": {
            "method": "Structural Topic Model",
            "data": "DeepSeek user reviews on Google Play",
            "prevalence_formula": "prevalence ~ score + period + score:period",
            "period_coding": {
                "0": "earlier period",
                "1": "later period",
            },
        },
        "Ip": {},
        "If": {},
        "Ma": {},
        "validation_warnings": {
            "Ip": [],
            "If": [],
            "Ma": [],
        },
    }

    print("\nRunning Ip — Interpreter...")
    ip_prompt = (
        "Analyze this research input. Focus on the strongest supported "
        "patterns and interpretations. Keep each item concise.\n\n"
        "=== INPUT ===\n"
        + human_input
    )

    try:
        ip, warnings = run_agent(
            "Ip",
            IP_SYSTEM,
            ip_prompt,
            IP_SCHEMA,
            lambda data, *_: validate_ip(data),
            (),
        )
        result["Ip"] = ip
        result["validation_warnings"]["Ip"] = warnings

        print("\nRunning If — Inferencer...")
        if_prompt = (
            "Critically evaluate the supplied STM material and Ip output. "
            "Generate concise, falsifiable hypotheses and rival explanations.\n\n"
            "=== ORIGINAL INPUT ===\n"
            + human_input[:4000]
            + "\n\n=== IP OUTPUT ===\n"
            + compact_json(ip)
        )

        inf, warnings = run_agent(
            "If",
            IF_SYSTEM,
            if_prompt,
            IF_SCHEMA,
            lambda data, *_: validate_if(data, ip),
            (),
        )
        result["If"] = inf
        result["validation_warnings"]["If"] = warnings

        print("\nRunning Ma — Meta agent...")
        ma_prompt = (
            "Audit the two agent outputs below. Identify the main risks and "
            "recommend a few high-value analytical and semantic next steps. "
            "Keep the entire response concise.\n\n"
            "=== IP OUTPUT ===\n"
            + compact_json(ip)
            + "\n\n=== IF OUTPUT ===\n"
            + compact_json(inf)
        )

        ma, warnings = run_agent(
            "Ma",
            MA_SYSTEM,
            ma_prompt,
            MA_SCHEMA,
            lambda data, *_: validate_ma(data, ip, inf),
            (),
        )
        result["Ma"] = ma
        result["validation_warnings"]["Ma"] = warnings

    except Exception as exc:
        result["pipeline_error"] = f"{type(exc).__name__}: {exc}"
        print(f"\nPIPELINE STOPPED: {exc}", file=sys.stderr)

    return result


def main():
    global MODEL

    parser = argparse.ArgumentParser(
        description="Three-agent STM analysis using Llama 3.1 70B."
    )
    parser.add_argument(
        "input",
        nargs="?",
        default=str(DEFAULT_INPUT),
        help="Input text file; defaults to outputs/input.txt.",
    )
    parser.add_argument(
        "-o",
        "--output",
        default=str(DEFAULT_JSON),
        help="JSON output path.",
    )
    parser.add_argument(
        "--report",
        default=str(DEFAULT_REPORT),
        help="Text report output path.",
    )
    parser.add_argument(
        "--model",
        default=MODEL,
        help="Model name served by the endpoint.",
    )

    args = parser.parse_args()
    MODEL = args.model

    if not VLLM_URL:
        sys.exit(
            "Set the LLM_URL environment variable to the chat completions "
            "endpoint of your model server."
        )

    if args.input == "-":
        human_input = sys.stdin.read()
    else:
        input_path = Path(args.input).expanduser()
        try:
            human_input = input_path.read_text(encoding="utf-8")
        except OSError as exc:
            sys.exit(f"Cannot read input file {input_path}: {exc}")

    if not human_input.strip():
        sys.exit("The input file is empty.")

    if len(human_input) > MAX_INPUT_CHARS:
        print(
            f"WARNING: input has {len(human_input)} characters. "
            f"Truncating to {MAX_INPUT_CHARS} characters. "
            "Check that important coefficients and topic results are retained.",
            file=sys.stderr,
        )
        human_input = human_input[:MAX_INPUT_CHARS]

    print("=" * 70)
    print("THREE-AGENT ANALYSIS")
    print(f"Model: {MODEL}")
    print(f"Input: {args.input}")
    print(f"Maximum completion tokens per call: {MAX_TOKENS}")
    print("=" * 70)

    result = run_pipeline(human_input)
    report = render_report(result)

    print("\n" + report)

    try:
        save_json(args.output, result)
        save_text(args.report, report)
    except OSError as exc:
        sys.exit(f"Could not save output files: {exc}")

    print("\n" + "=" * 70)
    print("OUTPUT FILES")
    print(f"JSON: {Path(args.output).expanduser()}")
    print(f"Text: {Path(args.report).expanduser()}")

    if result.get("pipeline_error"):
        print(
            "\nThe pipeline stopped before all agents completed. "
            "Partial results and the error were saved."
        )
    else:
        print("\nAll three agents completed.")

    print("=" * 70)


if __name__ == "__main__":
    main()