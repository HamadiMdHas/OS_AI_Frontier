import argparse
import json
import math
import os
import re
import sys
import textwrap
from pathlib import Path

import requests


OUTPUT_DIR = Path("outputs")

DEFAULT_INPUT = OUTPUT_DIR / "diag.txt"
DEFAULT_JSON = OUTPUT_DIR / "diagout.json"
DEFAULT_REPORT = OUTPUT_DIR / "diagout.txt"

VLLM_URL = os.environ.get("LLM_URL", "")
MODEL = os.environ.get("LLM_MODEL", "meta-llama/Llama-3.1-70B-Instruct")
API_KEY = os.environ.get("LLM_API_KEY", "")

CONTEXT_WINDOW = 8192
MAX_TOKENS = 2048
MIN_COMPLETION_TOKENS = 600
SAFETY_MARGIN = 150
TEMPLATE_OVERHEAD = 50
CHARS_PER_TOKEN = 3

REQUEST_TIMEOUT = 600
TEMPERATURE = 0.2
MAX_RETRIES = 1

MAX_INPUT_CHARS = 12000


CODE_DEFINITIONS = {
    "c1": (
        "Reviewers express enthusiasm, excitement, amazement, delight, or "
        "strong positive emotional reactions toward DeepSeek."
    ),
    "c2": (
        "Reviewers express concern, distrust, suspicion, or criticism about "
        "DeepSeek's legitimacy, privacy, data collection, security, "
        "transparency, censorship, terms, or trustworthiness."
    ),
    "c3": (
        "Reviewers compare DeepSeek with other AI tools, models, assistants, "
        "or generative AI companies. Comparisons can be positive, negative, "
        "or mixed and can concern price, quality, accuracy, capabilities, "
        "speed, or overall value."
    ),
    "c4": (
        "Reviewers explicitly report what they asked, prompted, instructed, "
        "or attempted to get DeepSeek to do, or explicitly describe a "
        "concrete task they tested it on. The task may succeed, fail, or "
        "partly succeed."
    ),
    "c5": (
        "Reviewers make propositions, expectations, suggestions, "
        "predictions, or hypothetical statements about future DeepSeek "
        "capabilities, development, or potential uses WITHOUT reporting "
        "that they explicitly asked DeepSeek to perform that task."
    ),
    "c6": (
        "Reviewers acknowledge a limitation, flaw, mistake, problem, or "
        "inconvenience in DeepSeek while nevertheless accepting, "
        "tolerating, excusing, or expressing optimism about the tool's "
        "overall value."
    ),
}

TOPIC_DESCRIPTIONS = {
    7: (
        "Users appreciate DeepSeek's free access, speed, simplicity, and "
        "powerful features without requiring payment."
    ),
    10: (
        "Users express amazement and gratitude for DeepSeek's helpfulness "
        "across studying, programming, and other everyday tasks."
    ),
    37: (
        "Users raise concerns about data collection, privacy policies, "
        "potential bias, and censorship related to Chinese political "
        "sensitivities."
    ),
    21: (
        "Users discuss information limits, outdated knowledge, internet "
        "access, trustworthiness, and censorship of sensitive topics."
    ),
    8: (
        "Users compare DeepSeek with ChatGPT in terms of speed, "
        "performance, and overall experience, expressing excitement or "
        "mixed reactions toward the alternatives."
    ),
    12: (
        "Users compare DeepSeek with Gemini, Copilot, Claude, and "
        "Perplexity, focusing on relative speed, capabilities, and "
        "competitiveness."
    ),
    11: (
        "Users discuss DeepSeek's ability to answer questions, explain "
        "concepts simply, and provide correct or incorrect factual "
        "information."
    ),
    24: (
        "Users report difficulties signing up, receiving verification "
        "codes, registering accounts, and completing email or OTP "
        "verification."
    ),
    23: (
        "Users request voice input and spoken responses, highlighting the "
        "desire for more natural conversational interaction and additional "
        "voice features."
    ),
    33: (
        "Users explore DeepSeek's ability to generate images and support "
        "creative tasks, while noting limitations in image creation and "
        "recognition."
    ),
    4: (
        "Users express hope for upcoming updates, new features, and "
        "improvements while waiting for the app to become available or "
        "perform better."
    ),
    32: (
        "Users discuss the app's current state, recent releases, and the "
        "need for timely improvements to resolve persistent problems."
    ),
}

NO_DESCRIPTION = "(no description supplied; do not infer its content)"


def build_codebook(topic_ids=None):
    lines = ["CODE DEFINITIONS (binary review-level codes, 1 = present):"]
    for key, text in CODE_DEFINITIONS.items():
        lines.append(f"{key}: {text}")

    if topic_ids is None:
        ids = sorted(TOPIC_DESCRIPTIONS)
    else:
        ids = sorted(topic_ids)

    lines.append("")
    lines.append("TOPIC DESCRIPTIONS (summaries of the STM topics):")
    for tid in ids:
        lines.append(
            f"Topic_{tid}: {TOPIC_DESCRIPTIONS.get(tid, NO_DESCRIPTION)}"
        )
    return "\n".join(lines)


STUDY_CONTEXT_TEMPLATE = """
Study: DeepSeek user reviews posted on Google Play (about 11,000 reviews).
A Structural Topic Model (STM) with 40 topics (Topic_1 ... Topic_40) was
fitted to the reviews. Six binary review-level codes, c1 to c6, are
correlated with each other and with the 40 topic variables. How the codes
were assigned, and the exact form of the topic variables, are not stated.

Input: Pearson correlations (phi for two 0/1 variables). "Estimate" or r is
the correlation coefficient, with its standard error (SE) and p-value.
Significance symbols: *** p<0.001, ** p<0.01, * p<0.05, dagger p<0.10.

{codebook}

Rules:
- The estimate is a correlation, not a regression coefficient.
- Codes are not mutually exclusive. A c-c correlation describes how often
  two codes co-occur in the same reviews.
- Topic rows are a SELECTED subset: per code, the significant topics with
  the highest estimates plus the most negative significant topic, as the
  selection rule in the input states. Topics not listed were not reported.
  Do not infer that unlisted topics are uncorrelated, and do not claim a
  listed topic is the strongest or only association beyond the rule.
- With about 11,000 reviews almost any non-trivial correlation is
  significant. Significance symbols and p-values alone do not show
  importance. Use the size and sign of r. Rough benchmarks only: about 0.1
  small, 0.3 moderate, 0.5 large.
- Topic descriptions summarize a topic. A correlation with a code does not
  show what any individual review says.
- For a topic without a supplied description, do not infer its content from
  its number or from the code it correlates with. State that it is missing.
- Cite numbers exactly as given in the input (rounding is acceptable).
  Do not invent statistics. Label any derived value as derived.
- Distinguish observations, interpretations, and hypotheses.
- Association does not establish causation or direction of influence.
- Ip must remain descriptive and must make no causal claims.
- State when important information is missing.
"""


def build_study_context(topic_ids=None):
    return STUDY_CONTEXT_TEMPLATE.format(
        codebook=build_codebook(topic_ids)
    )


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

Analyze the supplied correlation results using the code definitions and
topic descriptions.

Produce:
1. Salient observable patterns, numbered from 1. Each pattern must name the
   variables and quote their r value as given (for example c5~c6, r=0.3704).
   Cover both code-code (c-c) correlations and code-topic correlations.
2. Cautious interpretations supported by those patterns, numbered from 1.
   Relate each code's definition to the descriptions of its correlated
   topics. Note where they are semantically coherent, where they are not,
   and where a topic has no supplied description.
3. Relevant bodies of knowledge or research literatures.

Do not make causal claims. Use only the signs and significance shown in
the input. Do not invent results. Distinguish topic descriptions from actual
user intentions or experiences. Acknowledge missing information, including
the fact that only selected topics are shown.

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

Clearly distinguish hypotheses from established findings. Identify
alternative explanations, uncertainty, and additional data needed. Consider
in particular: the selection of topics shown, significance at large N,
overlap between codes, how well the code definitions and topic
descriptions capture what reviews say, and topics lacking descriptions.
Do not invent coefficients or claim unsupported statistical significance.
Cite existing pattern and interpretation IDs.

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
Robustness checks, multiple-comparison correction across the 240 code-topic
tests, confidence intervals, partial correlations or regressions that
account for correlated codes, code reliability and validity checks, topic
stability and topic-count sensitivity, and inspection of representative
reviews.

SEMANTIC_INTERVENTION:
Reading top words and representative reviews of topics without
descriptions, lemmatization, stopword changes, synonym merging, phrase
detection, topic splitting or pooling (for example topics that compare
DeepSeek with other tools), keyword-in-context inspection, entity masking,
and rerunning the model after removing dominant terms.

Choose steps relevant to the supplied results. Do not invent findings.
Keep the output concise and cite existing IDs where appropriate.

""" + TAG_RULES


_EST = r"-?\d+\.\d+"
_P = r"<0\.001|\d+\.\d+"
_SIG = r"(?:\*{1,3}|\u2020)?"
_PICK = r"(highest estimate|strongest negative)?"

ROW_CC = re.compile(
    r"^\s*(c\d)\s+(c\d)\s+(\d+)\s+"
    r"(" + _EST + r")\s+(" + _EST + r")\s+(" + _EST + r")\s+"
    r"(" + _P + r")\s*(" + _SIG + r")\s*$"
)

ROW_CT = re.compile(
    r"^\s*(c\d)\s+(Topic_\d+)\s+(\d+)\s+"
    r"(" + _EST + r")\s+(" + _EST + r")\s+(" + _EST + r")\s+"
    r"(" + _P + r")\s*(" + _SIG + r")\s*" + _PICK + r"\s*$"
)


def parse_diag(text):
    cc, ct = [], []
    selection_rule = legend = None

    for line in text.splitlines():
        stripped = line.strip()
        if stripped.startswith("Topic selection:"):
            selection_rule = stripped
            continue
        if stripped.startswith("Significance:"):
            legend = stripped
            continue

        m = ROW_CC.match(line)
        if m:
            cc.append({
                "a": m.group(1), "b": m.group(2), "n": m.group(3),
                "est": m.group(4), "se": m.group(5),
                "p": m.group(7), "sig": m.group(8) or "",
            })
            continue

        m = ROW_CT.match(line)
        if m:
            ct.append({
                "c": m.group(1), "topic": m.group(2), "n": m.group(3),
                "est": m.group(4), "se": m.group(5),
                "p": m.group(7), "sig": m.group(8) or "",
                "pick": m.group(9) or "",
            })

    if not cc and not ct:
        return None

    return {
        "cc": cc,
        "ct": ct,
        "selection_rule": selection_rule,
        "legend": legend,
    }


def _fmt_p(p):
    return "p" + p if p.startswith("<") else "p=" + p


def render_evidence(parsed):
    lines = [
        "Correlation results (Pearson r; phi for two 0/1 variables).",
    ]
    if parsed["selection_rule"]:
        lines.append("Selection rule stated in the file: "
                     + parsed["selection_rule"])
    if parsed["legend"]:
        lines.append(parsed["legend"])

    if parsed["cc"]:
        ns = sorted({r["n"] for r in parsed["cc"]})
        per_row = len(ns) > 1
        lines += ["", "[A] Correlations between codes (all pairs), N="
                  + "/".join(ns)]
        for r in parsed["cc"]:
            row = (f"{r['a']}~{r['b']}: r={r['est']}, SE={r['se']}, "
                   f"{_fmt_p(r['p'])}")
            if r["sig"]:
                row += " " + r["sig"]
            if per_row:
                row += f", N={r['n']}"
            lines.append(row)

    if parsed["ct"]:
        ns = sorted({r["n"] for r in parsed["ct"]})
        per_row = len(ns) > 1
        lines += ["", "[B] Selected topic correlations per code, N="
                  + "/".join(ns)]
        for r in parsed["ct"]:
            row = (f"{r['c']}~{r['topic']}: r={r['est']}, SE={r['se']}, "
                   f"{_fmt_p(r['p'])}")
            if r["sig"]:
                row += " " + r["sig"]
            if r["pick"]:
                row += f" [{r['pick']}]"
            if per_row:
                row += f", N={r['n']}"
            lines.append(row)

    return "\n".join(lines)


NUM_RE = re.compile(r"(?<![\w.])[-\u2212]?(\d*\.\d+)(?!\w)")
TOPIC_RE = re.compile(r"\bTopic[_ ](\d+)\b")

ALLOWED_CONSTANTS = {0.001, 0.01, 0.05, 0.1, 0.3, 0.5}


def build_grounding(evidence_text):
    numbers = {
        abs(float(m.group(1))) for m in NUM_RE.finditer(evidence_text)
    }
    topics = {int(t) for t in TOPIC_RE.findall(evidence_text)}
    return {"numbers": numbers | ALLOWED_CONSTANTS, "topics": topics}


def _number_supported(token, allowed):
    decimals = len(token.split(".")[1])
    x = float(token)
    scale = 10 ** decimals
    for v in allowed:
        if abs(round(v, decimals) - x) < 1e-9:
            return True
        if abs(math.floor(v * scale + 1e-9) / scale - x) < 1e-9:
            return True
    return False


def check_grounding(data, grounding):
    problems = []
    seen_numbers, seen_topics = set(), set()

    for text in all_strings(data):
        for m in NUM_RE.finditer(text):
            token = m.group(1)
            if token in seen_numbers:
                continue
            seen_numbers.add(token)
            if not _number_supported(token, grounding["numbers"]):
                problems.append(
                    f"Number {token} does not appear in the input. "
                    "Use values from the input or label derived values."
                )
        for t in TOPIC_RE.findall(text):
            t = int(t)
            if t not in grounding["topics"] and t not in seen_topics:
                seen_topics.add(t)
                problems.append(
                    f"Topic_{t} is not among the topics in the input."
                )

    return problems


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


class TruncatedOutput(RuntimeError):
    pass


def estimate_tokens(text):
    return len(text) // CHARS_PER_TOKEN + 1


def completion_budget(system, user):
    prompt_tokens = (
        estimate_tokens(system) + estimate_tokens(user) + TEMPLATE_OVERHEAD
    )
    available = CONTEXT_WINDOW - prompt_tokens - SAFETY_MARGIN

    if available < MIN_COMPLETION_TOKENS:
        raise RuntimeError(
            f"The prompt is estimated at {prompt_tokens} tokens, leaving "
            f"only {available} for the answer in a {CONTEXT_WINDOW}-token "
            f"window (minimum {MIN_COMPLETION_TOKENS}). Shorten the input "
            "or raise CONTEXT_WINDOW if the server allows it."
        )

    return min(MAX_TOKENS, available), prompt_tokens


def call_llm(system, user, schema):
    max_tokens, prompt_tokens = completion_budget(system, user)
    print(
        f"    prompt ~{prompt_tokens} tokens; "
        f"completion limit {max_tokens} tokens"
    )

    payload = {
        "model": MODEL,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        "temperature": TEMPERATURE,
        "max_tokens": max_tokens,
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
            f"Requested completion budget: {max_tokens} tokens in a "
            f"{CONTEXT_WINDOW}-token window. If this is a context-length "
            "error, check that CONTEXT_WINDOW matches the server's "
            "--max-model-len."
        )

    try:
        body = response.json()
        choice = body["choices"][0]
        content = choice["message"]["content"]
    except (ValueError, KeyError, IndexError, TypeError) as exc:
        raise RuntimeError(
            "Unexpected response format:\n" + response.text[:2000]
        ) from exc

    usage = body.get("usage") or {}
    if usage:
        print(
            f"    used: prompt {usage.get('prompt_tokens')}, "
            f"completion {usage.get('completion_tokens')}"
        )

    if choice.get("finish_reason") == "length":
        raise TruncatedOutput(
            f"Output was cut off at {max_tokens} tokens, so the JSON is "
            "incomplete."
        )

    try:
        result = json.loads(content)
    except ValueError as exc:
        raise RuntimeError(
            "Could not parse the model's JSON response:\n"
            + response.text[:2000]
        ) from exc

    if not isinstance(result, dict):
        raise RuntimeError("The model response is not a JSON object.")

    return result


def validate_ip(data, grounding):
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

    problems += check_grounding(data, grounding)
    return problems


def validate_if(data, ip, grounding):
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

    problems += check_grounding(data, grounding)
    return problems


def validate_ma(data, ip, inf, grounding):
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

    problems += check_grounding(data, grounding)
    return problems


def run_agent(name, system, prompt, schema, validator):
    last_data = None
    last_problems = []

    for attempt in range(MAX_RETRIES + 1):
        current_prompt = prompt

        if attempt:
            feedback = "; ".join(last_problems[:4])
            current_prompt += (
                "\n\nPlease correct these problems: "
                + feedback
                + ". Return the complete corrected JSON. Keep it concise."
            )

        print(f"[{name}] sending request...")

        try:
            last_data = call_llm(system, current_prompt, schema)
        except TruncatedOutput as exc:
            print(f"[{name}] {exc}", file=sys.stderr)
            if attempt == MAX_RETRIES:
                raise
            last_problems = [
                "the previous response was cut off by the token limit; "
                "return fewer, briefer items"
            ]
            continue

        try:
            last_problems = validator(last_data)
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
        "THREE-AGENT ANALYSIS: CODES c1-c6 AND STM TOPICS",
        f"LLM: {result['model']}",
        "Data: DeepSeek user reviews on Google Play",
        "Input: Pearson correlations among c1-c6 and between c1-c6 and "
        "selected topics",
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


def prepare_input(raw_text):
    parsed = parse_diag(raw_text)

    if parsed:
        evidence = render_evidence(parsed)
        mode = "parsed"
        print(
            f"Parsed {len(parsed['cc'])} c-c rows and "
            f"{len(parsed['ct'])} code-topic rows."
        )
    else:
        evidence = raw_text
        mode = "raw"
        print(
            "WARNING: could not parse the correlation tables; "
            "using the raw text.",
            file=sys.stderr,
        )

    grounding = build_grounding(evidence)
    return evidence, grounding, mode


def run_pipeline(raw_text):
    evidence, grounding, mode = prepare_input(raw_text)

    topic_ids = grounding["topics"] or None
    study_context = build_study_context(topic_ids)

    ip_system = study_context + "\n" + IP_SYSTEM
    if_system = study_context + "\n" + IF_SYSTEM
    ma_system = study_context + "\n" + MA_SYSTEM

    result = {
        "model": MODEL,
        "study_context": {
            "method": "Structural Topic Model (40 topics)",
            "data": "DeepSeek user reviews on Google Play",
            "statistic": "Pearson correlation (phi for 0/1 pairs)",
            "codes": CODE_DEFINITIONS,
            "topics_with_descriptions": sorted(
                t for t in (topic_ids or TOPIC_DESCRIPTIONS)
                if t in TOPIC_DESCRIPTIONS
            ),
            "topics_without_descriptions": sorted(
                t for t in (topic_ids or [])
                if t not in TOPIC_DESCRIPTIONS
            ),
        },
        "input_mode": mode,
        "input_evidence": evidence,
        "Ip": {},
        "If": {},
        "Ma": {},
        "validation_warnings": {
            "Ip": [],
            "If": [],
            "Ma": [],
        },
    }

    try:
        print("\nRunning Ip — Interpreter...")
        ip_prompt = (
            "Analyze this research input. Focus on the strongest supported "
            "patterns and interpretations. Keep each item concise.\n\n"
            "=== INPUT ===\n"
            + evidence
        )

        ip, warnings = run_agent(
            "Ip",
            ip_system,
            ip_prompt,
            IP_SCHEMA,
            lambda data: validate_ip(data, grounding),
        )
        result["Ip"] = ip
        result["validation_warnings"]["Ip"] = warnings

        print("\nRunning If — Inferencer...")
        if_prompt = (
            "Critically evaluate the supplied correlation results and Ip "
            "output. Generate concise, falsifiable hypotheses and rival "
            "explanations.\n\n"
            "=== ORIGINAL INPUT ===\n"
            + evidence
            + "\n\n=== IP OUTPUT ===\n"
            + compact_json(ip)
        )

        inf, warnings = run_agent(
            "If",
            if_system,
            if_prompt,
            IF_SCHEMA,
            lambda data: validate_if(data, ip, grounding),
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
            ma_system,
            ma_prompt,
            MA_SCHEMA,
            lambda data: validate_ma(data, ip, inf, grounding),
        )
        result["Ma"] = ma
        result["validation_warnings"]["Ma"] = warnings

    except Exception as exc:
        result["pipeline_error"] = f"{type(exc).__name__}: {exc}"
        print(f"\nPIPELINE STOPPED: {exc}", file=sys.stderr)

    return result


def main():
    global MODEL, CONTEXT_WINDOW

    parser = argparse.ArgumentParser(
        description="Three-agent analysis of c-topic correlations "
                    "using Llama 3.1 70B."
    )
    parser.add_argument(
        "input",
        nargs="?",
        default=str(DEFAULT_INPUT),
        help="Input text file; defaults to outputs/diag.txt.",
    )
    parser.add_argument(
        "-o",
        "--output",
        default=str(DEFAULT_JSON),
        help="JSON output path; defaults to outputs/diagout.json.",
    )
    parser.add_argument(
        "--report",
        default=str(DEFAULT_REPORT),
        help="Text report output path; defaults to outputs/diagout.txt.",
    )
    parser.add_argument(
        "--model",
        default=MODEL,
        help="Model name served by the endpoint.",
    )
    parser.add_argument(
        "--context-window",
        type=int,
        default=CONTEXT_WINDOW,
        help="Total context window of the served model (default 8192).",
    )

    args = parser.parse_args()
    MODEL = args.model
    CONTEXT_WINDOW = args.context_window

    if not VLLM_URL:
        sys.exit(
            "Set the LLM_URL environment variable to the chat completions "
            "endpoint of your model server."
        )

    if args.input == "-":
        raw_text = sys.stdin.read()
    else:
        input_path = Path(args.input).expanduser()
        try:
            raw_text = input_path.read_text(encoding="utf-8")
        except OSError as exc:
            sys.exit(f"Cannot read input file {input_path}: {exc}")

    if not raw_text.strip():
        sys.exit("The input file is empty.")

    if len(raw_text) > MAX_INPUT_CHARS:
        print(
            f"WARNING: input has {len(raw_text)} characters. "
            f"Truncating to {MAX_INPUT_CHARS} characters. "
            "Check that important results are retained.",
            file=sys.stderr,
        )
        raw_text = raw_text[:MAX_INPUT_CHARS]

    print("=" * 70)
    print("THREE-AGENT ANALYSIS")
    print(f"Model: {MODEL}")
    print(f"Input: {args.input}")
    print(f"Context window: {CONTEXT_WINDOW}; "
          f"max completion tokens per call: {MAX_TOKENS}")
    print("=" * 70)

    result = run_pipeline(raw_text)
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