import json
import re
import hashlib
import random
from pathlib import Path
from typing import Any, Dict, Optional, List, Tuple

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM


# ============================================================
# 基础工具
# ============================================================

def is_nonempty_text(x: Optional[Any]) -> bool:
    return isinstance(x, str) and x.strip() != ""


def normalize_text(x: Optional[str]) -> Optional[str]:
    if not isinstance(x, str):
        return None
    x = x.strip()
    return x if x else None


def clean_text_basic(text: str) -> str:
    if not isinstance(text, str):
        return ""
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = text.replace("\u00a0", " ")
    text = text.replace("\u200b", "")
    text = re.sub(r"[ \t]+\n", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def find_first(pattern: str, text: str, flags=0) -> Optional[str]:
    m = re.search(pattern, text, flags)
    if m:
        return m.group(1).strip()
    return None


def split_paragraphs(text: str) -> List[str]:
    return [x.strip() for x in re.split(r"\n\s*\n", text) if x.strip()]


def safe_json_loads(line: str) -> Optional[Dict[str, Any]]:
    try:
        return json.loads(line)
    except Exception:
        return None


def normalize_for_dedup(text: Optional[str]) -> str:
    if not isinstance(text, str):
        return ""
    x = text.lower().strip()
    x = re.sub(r"\$+", "", x)
    x = re.sub(r"\\boxed\s*\{([^}]*)\}", r"\1", x)
    x = re.sub(r"\\text\s*\{([^}]*)\}", r"\1", x)
    x = re.sub(r"\\[a-zA-Z]+", " ", x)
    x = re.sub(r"[^a-z0-9\u4e00-\u9fff]+", " ", x)
    x = re.sub(r"\s+", " ", x).strip()
    return x


def text_hash(text: str) -> str:
    return hashlib.md5(text.encode("utf-8")).hexdigest()


# ============================================================
# 噪声清理
# ============================================================

def remove_linewise_noise(text: str) -> str:
    lines = text.splitlines()
    cleaned = []

    noise_line_patterns = [
        r"^\s*\*?\*?\s*Date\s*:\s*.*$",
        r"^\s*\*?\*?\s*Author\s*:\s*.*$",
        r"^\s*\*?\*?\s*Published\s*:\s*.*$",
        r"^\s*\*?\*?\s*Updated\s*:\s*.*$",
        r"^\s*\*?\*?\s*Tags\s*:\s*.*$",
        r"^\s*\*?\*?\s*Source\s*:\s*.*$",
        r"^\s*\*?\*?\s*References?\s*:\s*.*$",
        r"^\s*---+\s*$",
        r"^\s*Advertisement\s*$",
        r"^\s*Share\s*$",
    ]

    for line in lines:
        if not any(re.match(p, line, flags=re.I) for p in noise_line_patterns):
            cleaned.append(line)

    text = "\n".join(cleaned)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


# ============================================================
# 题目/解答/答案抽取
# ============================================================

def extract_title(text: str) -> Optional[str]:
    patterns = [
        r"^\s*#\s*(.*?)\n",
        r"^\s*##\s*(.*?)\n",
        r"^\s*Title\s*:\s*(.*?)\n",
    ]
    for p in patterns:
        t = find_first(p, text, flags=re.M | re.S | re.I)
        if t:
            return t.strip()

    first_line = text.splitlines()[0].strip() if text.splitlines() else ""
    if 0 < len(first_line) <= 150:
        return first_line
    return None


def extract_topic(text: str, title: Optional[str] = None) -> Optional[str]:
    patterns = [
        r"###\s*Concept:\s*(.*?)(?=\n)",
        r"\*\*Topic:\*\*\s*(.*?)(?=\n|$)",
        r"\*\*Concept:\*\*\s*(.*?)(?=\n|$)",
        r"Chapter\s+\d+\s*:\s*(.*)",
    ]
    for p in patterns:
        x = find_first(p, text, flags=re.S | re.I)
        if x:
            return x.strip()

    if title and 0 < len(title) <= 120:
        return title.strip()

    return None


def extract_problem_generic(text: str) -> Optional[str]:
    text = text.strip()
    if not text:
        return None

    explicit_patterns = [
        r"\*\*The Problem:\*\*\s*(.*?)(?=\n\s*\*\*(?:My Solution|Solution|Answer|Explanation|Proof|解答|答案):\*\*|\Z)",
        r"\*\*Problem:\*\*\s*(.*?)(?=\n\s*\*\*(?:My Solution|Solution|Answer|Explanation|Proof|解答|答案):\*\*|\Z)",
        r"\*\*Question:\*\*\s*(.*?)(?=\n\s*\*\*(?:My Solution|Solution|Answer|Explanation|Proof|解答|答案):\*\*|\Z)",
        r"\*\*题目:\*\*\s*(.*?)(?=\n\s*\*\*(?:解答|答案|解析):\*\*|\Z)",
        r"\*\*问题:\*\*\s*(.*?)(?=\n\s*\*\*(?:解答|答案|解析):\*\*|\Z)",
        r"(?:^|\n)(?:###\s*)?Question\s+\d+\s*(.*?)(?=\n\s*\*\*(?:My Solution|Solution|Answer|Explanation|Proof):\*\*|\Z)",
    ]
    for p in explicit_patterns:
        x = find_first(p, text, flags=re.S | re.I)
        if x and len(x.strip()) > 10:
            return x.strip()

    chunks = split_paragraphs(text)
    candidate_chunks = chunks[:10]

    command_patterns = [
        r"\bSolve\b", r"\bFind\b", r"\bEvaluate\b", r"\bCalculate\b",
        r"\bDetermine\b", r"\bCompute\b", r"\bProve\b", r"\bConvert\b",
        r"\bDistinguish\b", r"求", r"计算", r"证明", r"解方程", r"求解",
    ]

    best_chunk = None
    best_score = -10

    for chunk in candidate_chunks:
        score = 0

        if any(re.search(p, chunk, flags=re.I) for p in command_patterns):
            score += 5
        if re.search(r"\\\[|\\\(|=|\d", chunk):
            score += 2
        if "?" in chunk or "？" in chunk:
            score += 2

        L = len(chunk)
        if 20 <= L <= 700:
            score += 2
        elif L <= 1200:
            score += 1

        if re.search(r"\bThus\b|\bTherefore\b|\bHence\b|所以|因此", chunk, flags=re.I):
            score -= 3
        if re.search(r"\*\*(?:My Solution|Solution|Answer|Explanation|Proof|解答|答案):\*\*", chunk, flags=re.I):
            score -= 4

        if score > best_score:
            best_score = score
            best_chunk = chunk

    if best_chunk and best_score >= 3:
        return best_chunk.strip()

    return None


def extract_solution_explicit(text: str) -> Optional[str]:
    patterns = [
        r"\*\*My Solution:\*\*\s*(.*?)(?=\n\s*\*\*(?:Clarification|Conclusion):\*\*|\Z)",
        r"\*\*Solution:\*\*\s*(.*?)(?=\n\s*\*\*(?:Clarification|Conclusion):\*\*|\Z)",
        r"\*\*Answer:\*\*\s*(.*?)(?=\n\s*\*\*(?:Clarification|Conclusion):\*\*|\Z)",
        r"\*\*Explanation:\*\*\s*(.*?)(?=\n\s*\*\*(?:Clarification|Conclusion):\*\*|\Z)",
        r"\*\*Proof:\*\*\s*(.*?)(?=\n\s*\*\*(?:Clarification|Conclusion):\*\*|\Z)",
        r"\*\*解答:\*\*\s*(.*?)(?=\n\s*\*\*(?:答案|结论|总结):\*\*|\Z)",
        r"\*\*解析:\*\*\s*(.*?)(?=\n\s*\*\*(?:答案|结论|总结):\*\*|\Z)",
    ]
    for p in patterns:
        block = find_first(p, text, flags=re.S | re.I)
        if block:
            return block.strip()

    tail_patterns = [
        r"\*\*My Solution:\*\*\s*(.*)$",
        r"\*\*Solution:\*\*\s*(.*)$",
        r"\*\*Answer:\*\*\s*(.*)$",
        r"\*\*Explanation:\*\*\s*(.*)$",
        r"\*\*Proof:\*\*\s*(.*)$",
        r"\*\*解答:\*\*\s*(.*)$",
        r"\*\*解析:\*\*\s*(.*)$",
    ]
    for p in tail_patterns:
        block = find_first(p, text, flags=re.S | re.I)
        if block:
            return block.strip()

    return None


def extract_solution_fallback(text: str) -> Optional[str]:
    chunks = split_paragraphs(text)
    if not chunks:
        return None

    scored = []
    for i, chunk in enumerate(chunks):
        score = 0
        if re.search(r"\bThus\b|\bTherefore\b|\bHence\b|所以|因此", chunk, flags=re.I):
            score += 3
        if re.search(r"\\\[|\\\(|=", chunk):
            score += 2
        if re.search(r"\bStep\b|\bCase\b|\bSolution\b|\bExplanation\b|证明|解|步骤", chunk, flags=re.I):
            score += 2
        if i >= len(chunks) // 2:
            score += 1
        if len(chunk) >= 40:
            score += 1
        scored.append((score, i, chunk))

    scored.sort(reverse=True)
    if scored and scored[0][0] >= 2:
        return scored[0][2]
    return None


def clean_candidate_answer(ans: str) -> str:
    ans = ans.strip()
    ans = re.sub(r"^\s*[:：\-–]\s*", "", ans)
    ans = re.sub(r"\s+", " ", ans).strip()
    ans = re.sub(r"^[\"'“”‘’]+|[\"'“”‘’]+$", "", ans)
    return ans


def is_plausible_final_answer(ans: str) -> bool:
    if not ans:
        return False
    if len(ans) > 220:
        return False
    bad_patterns = [
        r"\bwe will\b",
        r"\blet us\b",
        r"\bconsider\b",
        r"\bfirst\b.*\bthen\b",
        r"\bproof\b",
        r"\bexplanation\b",
    ]
    if any(re.search(p, ans, flags=re.I) for p in bad_patterns):
        return False
    return True


def extract_final_answer(solution: Optional[str], full_text: str) -> Tuple[Optional[str], str]:
    """
    返回 (final_answer, confidence)
    confidence in {"high", "medium", "low", "none"}
    """
    texts = [x for x in [solution, full_text] if x]

    strong_patterns = [
        r"\*\*Conclusion:\*\*\s*(.{1,180}?)(?:\n\n|\Z)",
        r"\*\*答案:\*\*\s*(.{1,180}?)(?:\n\n|\Z)",
        r"\*\*最终答案:\*\*\s*(.{1,180}?)(?:\n\n|\Z)",
        r"\\boxed\{([^{}]{1,120})\}",
        r"\bThe answer is\s*([^\n\.]{1,180})",
        r"\bTherefore,?\s*([^\n]{1,180})",
        r"\bHence,?\s*([^\n]{1,180})",
        r"\bThus,?\s*([^\n]{1,180})",
        r"因此，?\s*([^\n]{1,180})",
        r"所以，?\s*([^\n]{1,180})",
    ]

    weak_patterns = [
        r"\bis approximately\s*([^\.\n]{1,60})",
        r"\bis\s*([\-+]?\d+(?:\.\d+)?)\s*$",
        r"\bfinal answer\s*[:：]?\s*(.{1,120}?)(?:\n|$)",
    ]

    candidates_high = []
    candidates_mid = []

    for txt in texts:
        for p in strong_patterns:
            for m in re.finditer(p, txt, flags=re.I | re.S):
                cand = clean_candidate_answer(m.group(1))
                if is_plausible_final_answer(cand):
                    candidates_high.append(cand)

        for p in weak_patterns:
            for m in re.finditer(p, txt, flags=re.I | re.S):
                cand = clean_candidate_answer(m.group(1))
                if is_plausible_final_answer(cand):
                    candidates_mid.append(cand)

    if candidates_high:
        # 优先选择较长的答案（通常更完整），长度相同时选字母顺序靠前的
        candidates_high = sorted(set(candidates_high), key=lambda x: (-len(x), x))
        return candidates_high[0], "high"

    if candidates_mid:
        candidates_mid = sorted(set(candidates_mid), key=lambda x: (-len(x), x))
        return candidates_mid[0], "medium"

    if solution:
        sentences = [x.strip() for x in re.split(r"(?<=[.!?。！？])\s+|\n{2,}", solution) if x.strip()]
        if sentences:
            last = clean_candidate_answer(sentences[-1])
            if is_plausible_final_answer(last):
                return last, "low"

    return None, "none"


# ============================================================
# 长文本 smarter 截取
# ============================================================

def smart_preview(text: str, head_chars: int = 900, tail_chars: int = 700, key_para_limit: int = 2) -> str:
    text = text.strip()
    if len(text) <= head_chars + tail_chars + 200:
        return text

    head = text[:head_chars]
    tail = text[-tail_chars:]

    paras = split_paragraphs(text)
    scored_paras = []

    keywords = [
        r"\bSolution\b", r"\bAnswer\b", r"\bExplanation\b", r"\bProof\b",
        r"\bTherefore\b", r"\bHence\b", r"\bThus\b",
        r"解答", r"答案", r"解析", r"因此", r"所以", r"证明",
    ]

    for i, para in enumerate(paras):
        score = 0
        for kw in keywords:
            if re.search(kw, para, flags=re.I):
                score += 2
        if re.search(r"\\boxed|=", para):
            score += 1
        if len(para) >= 40:
            score += 1
        scored_paras.append((score, i, para))

    scored_paras.sort(reverse=True)
    chosen = []
    for score, _, para in scored_paras[:key_para_limit]:
        if score > 0:
            chosen.append(para)

    merged = head + "\n\n[...snip...]\n\n"
    if chosen:
        merged += "\n\n".join(chosen) + "\n\n[...snip...]\n\n"
    merged += tail
    return merged


# ============================================================
# 模型兜底抽取
# ============================================================

class QwenExtractor:
    def __init__(self, model_path: str):
        self.model_path = model_path
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            device_map="auto",
            trust_remote_code=True,
        )

    def _build_prompt(self, text: str) -> str:
        return f"""You are an information extraction system.

Extract from the following math-related text these fields:
1. topic
2. problem
3. solution
4. final_answer

Rules:
- Return ONLY valid JSON.
- If a field cannot be found, use null.
- Keep problem as the original problem statement as much as possible.
- Keep solution as the main reasoning/derivation, not comments or references.
- Keep final_answer short and precise.
- Do not hallucinate.
- Prefer exact wording from the original text.

Output format:
{{
  "topic": ...,
  "problem": ...,
  "solution": ...,
  "final_answer": ...
}}

Text:
{text}
"""

    def _extract_json(self, generated: str) -> Dict[str, Any]:
        m = re.search(r"\{.*\}", generated, flags=re.S)
        if not m:
            return {"topic": None, "problem": None, "solution": None, "final_answer": None}
        try:
            obj = json.loads(m.group(0))
            return {
                "topic": normalize_text(obj.get("topic")),
                "problem": normalize_text(obj.get("problem")),
                "solution": normalize_text(obj.get("solution")),
                "final_answer": normalize_text(obj.get("final_answer")),
            }
        except Exception:
            return {"topic": None, "problem": None, "solution": None, "final_answer": None}

    @torch.inference_mode()
    def extract(self, text: str, max_new_tokens: int = 384) -> Dict[str, Any]:
        prompt = self._build_prompt(text)

        messages = [{"role": "user", "content": prompt}]
        if hasattr(self.tokenizer, "apply_chat_template"):
            model_input = self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True
            )
        else:
            model_input = prompt

        inputs = self.tokenizer(
            model_input,
            return_tensors="pt",
            truncation=True,
            max_length=2048,
        ).to(self.model.device)

        outputs = self.model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            temperature=0.0,
            pad_token_id=self.tokenizer.eos_token_id
        )

        generated = self.tokenizer.decode(
            outputs[0][inputs["input_ids"].shape[1]:],
            skip_special_tokens=True
        )
        return self._extract_json(generated)


# ============================================================
# 质量评估
# ============================================================

def compute_quality_score(record: Dict[str, Any]) -> Tuple[int, List[str]]:
    score = 0
    reasons = []

    problem = record.get("problem") or ""
    solution = record.get("solution") or ""
    final_answer = record.get("final_answer") or ""
    conf = record.get("final_answer_confidence", "none")

    if is_nonempty_text(problem):
        score += 2
    else:
        reasons.append("problem_missing")

    if is_nonempty_text(solution):
        score += 2
    else:
        reasons.append("solution_missing")

    if is_nonempty_text(final_answer):
        score += 2
    else:
        reasons.append("final_answer_missing")

    if len(problem) >= 20:
        score += 1
    else:
        reasons.append("problem_too_short")

    if len(solution) >= 40:
        score += 1
    else:
        reasons.append("solution_too_short")

    if 1 <= len(final_answer) <= 120:
        score += 1
    else:
        reasons.append("final_answer_length_abnormal")

    if conf == "high":
        score += 2
    elif conf == "medium":
        score += 1
    elif conf == "low":
        reasons.append("final_answer_low_confidence")
    else:
        reasons.append("final_answer_no_confidence")

    if problem and final_answer and normalize_for_dedup(problem) == normalize_for_dedup(final_answer):
        score -= 2
        reasons.append("problem_equals_answer")

    if solution and final_answer and normalize_for_dedup(solution) == normalize_for_dedup(final_answer):
        score -= 2
        reasons.append("solution_equals_answer")

    return score, reasons


def quality_bucket(score: int) -> str:
    if score >= 9:
        return "high"
    if score >= 6:
        return "medium"
    return "low"


# ============================================================
# 难度 / 类型分桶
# ============================================================

def infer_math_bucket(problem: Optional[str], solution: Optional[str]) -> str:
    text = f"{problem or ''}\n{solution or ''}"

    if re.search(r"\bprove\b|证明", text, flags=re.I):
        return "proof"
    if re.search(r"\bintegral\b|\\int|积分", text, flags=re.I):
        return "calculus"
    if re.search(r"\bmatrix\b|行列式|矩阵", text, flags=re.I):
        return "linear_algebra"
    if re.search(r"\bprobability\b|\bexpectation\b|概率|期望", text, flags=re.I):
        return "probability"
    if re.search(r"\btriangle\b|\bcircle\b|几何|角|边长", text, flags=re.I):
        return "geometry"
    if re.search(r"\bsolve\b.*\bx\b|方程|不等式|polynomial|代数", text, flags=re.I):
        return "algebra"
    return "general_math"


def infer_difficulty(problem: Optional[str], solution: Optional[str]) -> str:
    plen = len(problem or "")
    slen = len(solution or "")
    text = f"{problem or ''}\n{solution or ''}"

    complexity = 0
    complexity += 1 if plen > 80 else 0
    complexity += 1 if slen > 150 else 0
    complexity += 1 if re.search(r"\bcase\b|\bstep\b|因此|所以|首先|然后", text, flags=re.I) else 0
    complexity += 1 if re.search(r"\\int|\\sum|\\prod|\\frac|\\sqrt|\^|_", text) else 0

    if complexity >= 3:
        return "hard"
    if complexity >= 1:
        return "medium"
    return "easy"


# ============================================================
# 规则抽取主函数
# ============================================================

def parse_rule_based(sample: Dict[str, Any]) -> Dict[str, Any]:
    raw_text = sample.get("text", "")
    text = clean_text_basic(raw_text)
    text = remove_linewise_noise(text)

    title = extract_title(text)
    topic = extract_topic(text, title)
    problem = extract_problem_generic(text)
    solution = extract_solution_explicit(text)
    if not solution:
        solution = extract_solution_fallback(text)

    final_answer, final_answer_conf = extract_final_answer(solution, text)

    return {
        "id": sample.get("id"),
        "title": normalize_text(title),
        "topic": normalize_text(topic),
        "problem": normalize_text(problem),
        "solution": normalize_text(solution),
        "final_answer": normalize_text(final_answer),
        "final_answer_confidence": final_answer_conf,
        "metadata": sample.get("metadata", {}),
        "raw_text_preview": smart_preview(text),
    }


def merge_rule_and_model(rule_out: Dict[str, Any], model_out: Dict[str, Any]) -> Dict[str, Any]:
    merged = dict(rule_out)

    for field in ["topic", "problem", "solution", "final_answer"]:
        if not is_nonempty_text(merged.get(field)) and is_nonempty_text(model_out.get(field)):
            merged[field] = normalize_text(model_out.get(field))

    if not is_nonempty_text(merged.get("final_answer")):
        merged["final_answer_confidence"] = "none"
    elif merged.get("final_answer_confidence") in [None, "none"]:
        merged["final_answer_confidence"] = "medium"

    return merged


def needs_model_fallback(rule_out: Dict[str, Any]) -> bool:
    required = [
        is_nonempty_text(rule_out.get("problem")),
        is_nonempty_text(rule_out.get("solution")),
        is_nonempty_text(rule_out.get("final_answer")),
    ]
    return not all(required)


# ============================================================
# 抽检样本导出
# ============================================================

def export_review_samples(records: List[Dict[str, Any]], review_path: Path, sample_size: int = 100) -> None:
    if not records:
        review_path.write_text("", encoding="utf-8")
        return

    sample_size = min(sample_size, len(records))
    sampled = random.sample(records, sample_size)

    with review_path.open("w", encoding="utf-8") as f:
        for item in sampled:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")


# ============================================================
# 主流程
# ============================================================

def clean_with_model(
    input_path: str,
    output_path: str,
    stats_path: str,
    review_path: str,
    model_path: str,
    max_samples: Optional[int] = None,
    min_quality_score: int = 6,
    review_sample_size: int = 100,
):
    extractor = QwenExtractor(model_path)

    input_path = Path(input_path)
    output_path = Path(output_path)
    stats_path = Path(stats_path)
    review_path = Path(review_path)

    stats = {
        "total": 0,
        "json_parse_failed": 0,
        "rule_complete": 0,
        "model_called": 0,
        "final_complete": 0,
        "dropped_low_quality": 0,
        "dedup_exact_problem_removed": 0,
        "dedup_normalized_problem_removed": 0,
        "kept_records": 0,
        "bucket_counts": {},
        "difficulty_counts": {},
        "quality_bucket_counts": {},
    }

    seen_problem_exact = set()
    seen_problem_norm = set()
    review_pool = []

    with input_path.open("r", encoding="utf-8") as fin, output_path.open("w", encoding="utf-8") as fout:
        for idx, line in enumerate(fin):
            if max_samples is not None and idx >= max_samples:
                break

            sample = safe_json_loads(line)
            if sample is None:
                stats["json_parse_failed"] += 1
                continue

            stats["total"] += 1

            rule_out = parse_rule_based(sample)

            if (
                is_nonempty_text(rule_out["problem"])
                and is_nonempty_text(rule_out["solution"])
                and is_nonempty_text(rule_out["final_answer"])
            ):
                stats["rule_complete"] += 1
                merged = rule_out
                merged["extraction_source"] = "rule"
            else:
                stats["model_called"] += 1
                model_out = extractor.extract(rule_out["raw_text_preview"])
                merged = merge_rule_and_model(rule_out, model_out)
                merged["extraction_source"] = "rule+model"

                # 模型补完后，再试一次基于完整字段抽答案
                if not is_nonempty_text(merged.get("final_answer")) and is_nonempty_text(merged.get("solution")):
                    fa, conf = extract_final_answer(merged["solution"], rule_out["raw_text_preview"])
                    merged["final_answer"] = normalize_text(fa)
                    merged["final_answer_confidence"] = conf

            if (
                is_nonempty_text(merged.get("problem"))
                and is_nonempty_text(merged.get("solution"))
                and is_nonempty_text(merged.get("final_answer"))
            ):
                stats["final_complete"] += 1

            # 质量评估
            q_score, q_reasons = compute_quality_score(merged)
            merged["quality_score"] = q_score
            merged["quality_reasons"] = q_reasons
            merged["quality_bucket"] = quality_bucket(q_score)

            # 数学分桶
            merged["math_bucket"] = infer_math_bucket(merged.get("problem"), merged.get("solution"))
            merged["difficulty"] = infer_difficulty(merged.get("problem"), merged.get("solution"))

            # 低质量直接过滤
            if q_score < min_quality_score:
                stats["dropped_low_quality"] += 1
                if merged["quality_bucket"] == "low":
                    review_pool.append(merged)
                continue

            # 去重：problem 精确去重
            problem = merged.get("problem") or ""
            p_exact = problem.strip()
            if p_exact in seen_problem_exact:
                stats["dedup_exact_problem_removed"] += 1
                continue
            seen_problem_exact.add(p_exact)

            # 去重：problem 归一化去重
            p_norm = normalize_for_dedup(problem)
            if p_norm in seen_problem_norm:
                stats["dedup_normalized_problem_removed"] += 1
                continue
            seen_problem_norm.add(p_norm)

            # 更新统计
            stats["kept_records"] += 1
            stats["bucket_counts"][merged["math_bucket"]] = stats["bucket_counts"].get(merged["math_bucket"], 0) + 1
            stats["difficulty_counts"][merged["difficulty"]] = stats["difficulty_counts"].get(merged["difficulty"], 0) + 1
            stats["quality_bucket_counts"][merged["quality_bucket"]] = stats["quality_bucket_counts"].get(merged["quality_bucket"], 0) + 1

            fout.write(json.dumps(merged, ensure_ascii=False) + "\n")

            # 抽检池：优先收集边缘样本和 model 参与样本
            if merged["extraction_source"] == "rule+model" or merged["quality_bucket"] == "medium":
                review_pool.append(merged)

    export_review_samples(review_pool, review_path, review_sample_size)

    with stats_path.open("w", encoding="utf-8") as fstats:
        json.dump(stats, fstats, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, required=True, help="原始 jsonl 文件")
    parser.add_argument("--output", type=str, default="cleaned_with_model_v2.jsonl", help="清洗后的输出")
    parser.add_argument("--stats", type=str, default="clean_with_model_v2_stats.json", help="统计文件")
    parser.add_argument("--review", type=str, default="review_samples.jsonl", help="人工抽检样本")
    parser.add_argument("--model_path", type=str, default="/home/ubuntu/models/Qwen3-0.6B-Base", help="本地模型路径")
    parser.add_argument("--max_samples", type=int, default=None, help="调试时只跑前 N 条")
    parser.add_argument("--min_quality_score", type=int, default=6, help="最低质量分，低于则过滤")
    parser.add_argument("--review_sample_size", type=int, default=100, help="导出抽检样本数")

    args = parser.parse_args()

    clean_with_model(
        input_path=args.input,
        output_path=args.output,
        stats_path=args.stats,
        review_path=args.review,
        model_path=args.model_path,
        max_samples=args.max_samples,
        min_quality_score=args.min_quality_score,
        review_sample_size=args.review_sample_size,
    )