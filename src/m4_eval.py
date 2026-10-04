from __future__ import annotations

"""Module 4: RAGAS Evaluation — 4 metrics + failure analysis."""

import os, sys, json, math
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")
from dataclasses import dataclass

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import TEST_SET_PATH

METRICS = ("faithfulness", "answer_relevancy", "context_precision", "context_recall")


@dataclass
class EvalResult:
    question: str
    answer: str
    contexts: list[str]
    ground_truth: str
    faithfulness: float
    answer_relevancy: float
    context_precision: float
    context_recall: float


def load_test_set(path: str = TEST_SET_PATH) -> list[dict]:
    """Load test set from JSON. (Đã implement sẵn)"""
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def _score(value) -> float:
    """RAGAS trả NaN khi một metric không tính được; coi là 0.0 để báo cáo luôn là số hợp lệ."""
    try:
        value = float(value)
    except (TypeError, ValueError):
        return 0.0
    return 0.0 if math.isnan(value) else value


def evaluate_ragas(questions: list[str], answers: list[str],
                   contexts: list[list[str]], ground_truths: list[str]) -> dict:
    """Run RAGAS evaluation.

    RAGAS cần OPENAI_API_KEY (LLM judge + embeddings). Thiếu key hoặc RAGAS lỗi thì trả về
    điểm 0.0 và in lý do — KHÔNG bịa điểm.
    """
    zeros = {m: 0.0 for m in METRICS}
    zeros["per_question"] = []
    try:
        from ragas import evaluate
        from ragas.metrics import faithfulness, answer_relevancy, context_precision, context_recall
        from datasets import Dataset

        dataset = Dataset.from_dict({
            "question": questions, "answer": answers,
            "contexts": contexts, "ground_truth": ground_truths,
        })
        result = evaluate(dataset, metrics=[faithfulness, answer_relevancy,
                                            context_precision, context_recall])
        df = result.to_pandas()
        per_question = [
            EvalResult(
                question=row["question"], answer=row["answer"],
                contexts=list(row["contexts"]), ground_truth=row["ground_truth"],
                faithfulness=_score(row.get("faithfulness", 0.0)),
                answer_relevancy=_score(row.get("answer_relevancy", 0.0)),
                context_precision=_score(row.get("context_precision", 0.0)),
                context_recall=_score(row.get("context_recall", 0.0)),
            )
            for _, row in df.iterrows()
        ]
        aggregate = {m: _score(df[m].mean()) if m in df else 0.0 for m in METRICS}
        return {**aggregate, "per_question": per_question}
    except Exception as e:
        print(f"  ⚠️  RAGAS evaluation failed: {e}")
        return zeros


_DIAGNOSTIC_TREE = {
    "faithfulness": ("LLM hallucinating", "Tighten prompt, lower temperature"),
    "context_recall": ("Missing relevant chunks", "Improve chunking or add BM25"),
    "context_precision": ("Too many irrelevant chunks", "Add reranking or metadata filter"),
    "answer_relevancy": ("Answer doesn't match question", "Improve prompt template"),
}


def failure_analysis(eval_results: list[EvalResult], bottom_n: int = 10) -> list[dict]:
    """Analyze bottom-N worst questions using Diagnostic Tree."""
    scored = []
    for r in eval_results:
        values = {m: getattr(r, m) for m in METRICS}
        worst_metric = min(values, key=values.get)  # chỉ số thấp nhất của câu hỏi này
        scored.append((sum(values.values()) / len(values), worst_metric, values[worst_metric], r))

    scored.sort(key=lambda item: item[0])  # trung bình tăng dần: tệ nhất trước
    failures = []
    for avg, worst_metric, worst_score, r in scored[:bottom_n]:
        diagnosis, fix = _DIAGNOSTIC_TREE[worst_metric]
        failures.append({
            "question": r.question,
            "answer": r.answer,
            "ground_truth": r.ground_truth,
            "avg_score": round(avg, 4),
            "worst_metric": worst_metric,
            "score": round(worst_score, 4),
            "diagnosis": diagnosis,
            "suggested_fix": fix,
        })
    return failures


def save_report(results: dict, failures: list[dict], path: str = "reports/ragas_report.json"):
    """Save evaluation report to JSON. (Đã implement sẵn)"""
    parent_dir = os.path.dirname(path)
    if parent_dir:
        os.makedirs(parent_dir, exist_ok=True)
    report = {
        "aggregate": {k: v for k, v in results.items() if k != "per_question"},
        "num_questions": len(results.get("per_question", [])),
        "failures": failures,
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(f"Report saved to {path}")


if __name__ == "__main__":
    test_set = load_test_set()
    print(f"Loaded {len(test_set)} test questions")
    print("Run pipeline.py first to generate answers, then call evaluate_ragas().")
