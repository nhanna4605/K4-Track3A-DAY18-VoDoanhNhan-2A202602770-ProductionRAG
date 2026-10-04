from __future__ import annotations

"""Production RAG Pipeline — Ghép toàn bộ M1+M2+M3+M4+M5."""

import os, sys, time
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.m1_chunking import load_documents, chunk_hierarchical
from src.m2_search import HybridSearch
from src.m3_rerank import CrossEncoderReranker
from src.m4_eval import load_test_set, evaluate_ragas, failure_analysis, save_report
from src.m5_enrichment import enrich_chunks
from config import RERANK_TOP_K

# (source, parent_id) -> văn bản đoạn cha. Tìm kiếm khớp trên đoạn CON (chính xác),
# nhưng gửi đoạn CHA cho LLM (đủ ngữ cảnh) — đúng thiết kế Hierarchical của M1.
PARENT_TEXTS: dict[tuple[str, str], str] = {}

# Thời gian từng bước (giây) và từng truy vấn (ms) -> reports/latency_report.json
TIMINGS: dict = {"steps_s": {}, "query_ms": {"search": [], "rerank": [], "llm": []}}


def build_pipeline():
    """Build production RAG pipeline."""
    print("=" * 60)
    print("PRODUCTION RAG PIPELINE")
    print("=" * 60, flush=True)

    # Step 1: Load & Chunk (M1)
    t0 = time.time()
    print("\n[1/4] Chunking documents...", flush=True)
    docs = load_documents()
    all_chunks = []
    PARENT_TEXTS.clear()
    for doc in docs:
        parents, children = chunk_hierarchical(doc["text"], metadata=doc["metadata"])
        for parent in parents:
            PARENT_TEXTS[(doc["metadata"]["source"], parent.metadata["parent_id"])] = parent.text
        for child in children:
            all_chunks.append({"text": child.text, "metadata": {**child.metadata, "parent_id": child.parent_id}})
    TIMINGS["steps_s"]["1_chunking"] = round(time.time() - t0, 2)
    print(f"  ✓ {len(all_chunks)} chunks from {len(docs)} documents ({time.time()-t0:.1f}s)", flush=True)

    # Step 2: Enrichment (M5)
    t0 = time.time()
    print(f"\n[2/4] Enriching {len(all_chunks)} chunks (M5, 1 API call/chunk)...", flush=True)
    enriched = enrich_chunks(all_chunks)
    TIMINGS["steps_s"]["2_enrichment"] = round(time.time() - t0, 2)
    if enriched:
        all_chunks = [{"text": e.enriched_text, "metadata": e.auto_metadata} for e in enriched]
        print(f"  ✓ Enriched {len(enriched)} chunks ({time.time()-t0:.1f}s)", flush=True)
    else:
        print("  ⚠️  M5 not implemented — using raw chunks", flush=True)

    # Step 3: Index (M2)
    t0 = time.time()
    print(f"\n[3/4] Indexing {len(all_chunks)} chunks (BM25 + Dense)...", flush=True)
    search = HybridSearch()
    search.index(all_chunks)
    TIMINGS["steps_s"]["3_indexing"] = round(time.time() - t0, 2)
    print(f"  ✓ Indexed ({time.time()-t0:.1f}s)", flush=True)

    # Step 4: Reranker (M3)
    t0 = time.time()
    print("\n[4/4] Loading reranker...", flush=True)
    reranker = CrossEncoderReranker()
    TIMINGS["steps_s"]["4_reranker_load"] = round(time.time() - t0, 2)
    print(f"  ✓ Reranker ready ({time.time()-t0:.1f}s)", flush=True)

    return search, reranker


def run_query(query: str, search: HybridSearch, reranker: CrossEncoderReranker) -> tuple[str, list[str]]:
    """Run single query through pipeline."""
    t0 = time.perf_counter()
    results = search.search(query)
    docs = [{"text": r.text, "score": r.score, "metadata": r.metadata} for r in results]
    t1 = time.perf_counter()
    reranked = reranker.rerank(query, docs, top_k=RERANK_TOP_K)
    t2 = time.perf_counter()
    TIMINGS["query_ms"]["search"].append((t1 - t0) * 1000)
    TIMINGS["query_ms"]["rerank"].append((t2 - t1) * 1000)
    contexts = []
    for r in (reranked if reranked else results[:RERANK_TOP_K]):
        key = (r.metadata.get("source"), r.metadata.get("parent_id"))
        text = PARENT_TEXTS.get(key, r.text)  # đoạn cha nếu có, ngược lại giữ đoạn con
        if text not in contexts:  # nhiều đoạn con có thể cùng một cha
            contexts.append(text)

    from config import OPENAI_API_KEY
    t3 = time.perf_counter()
    if OPENAI_API_KEY and contexts:
        try:
            from openai import OpenAI
            client = OpenAI()
            context_str = "\n\n".join(contexts)
            resp = client.chat.completions.create(model="gpt-4o-mini", messages=[
                {"role": "system", "content": "Trả lời CHỈ dựa trên context. Nếu không có → nói 'Không tìm thấy.'"},
                {"role": "user", "content": f"Context:\n{context_str}\n\nCâu hỏi: {query}"},
            ])
            answer = resp.choices[0].message.content
        except Exception as e:
            print(f"  ⚠️  LLM generation failed: {e}", flush=True)
            answer = contexts[0]
    else:
        answer = contexts[0] if contexts else "Không tìm thấy thông tin."
    TIMINGS["query_ms"]["llm"].append((time.perf_counter() - t3) * 1000)
    return answer, contexts


def evaluate_pipeline(search: HybridSearch, reranker: CrossEncoderReranker):
    """Run evaluation on test set."""
    test_set = load_test_set()
    print(f"\n[Eval] Running {len(test_set)} queries...", flush=True)
    questions, answers, all_contexts, ground_truths = [], [], [], []

    for i, item in enumerate(test_set):
        answer, contexts = run_query(item["question"], search, reranker)
        questions.append(item["question"])
        answers.append(answer)
        all_contexts.append(contexts)
        ground_truths.append(item["ground_truth"])
        print(f"  [{i+1}/{len(test_set)}] {item['question'][:50]}...", flush=True)

    t0 = time.time()
    print(f"\n[Eval] Running RAGAS (4 metrics × {len(test_set)} questions)...", flush=True)
    results = evaluate_ragas(questions, answers, all_contexts, ground_truths)
    print(f"  ✓ RAGAS done ({time.time()-t0:.1f}s)", flush=True)

    print("\n" + "=" * 60)
    print("PRODUCTION RAG SCORES")
    print("=" * 60)
    for m in ["faithfulness", "answer_relevancy", "context_precision", "context_recall"]:
        s = results.get(m, 0)
        print(f"  {'✓' if s >= 0.75 else '✗'} {m}: {s:.4f}")

    failures = failure_analysis(results.get("per_question", []))
    save_report(results, failures)
    _save_details(results.get("per_question", []))
    return results


def _save_details(per_question) -> None:
    """Lưu điểm + câu trả lời + ngữ cảnh từng câu (phục vụ failure analysis) và bảng latency."""
    import json
    from dataclasses import asdict

    os.makedirs("reports", exist_ok=True)
    with open("reports/ragas_details.json", "w", encoding="utf-8") as f:
        json.dump([asdict(r) for r in per_question], f, ensure_ascii=False, indent=2)

    q = TIMINGS["query_ms"]
    avg = {k: round(sum(v) / len(v), 1) if v else 0.0 for k, v in q.items()}
    report = {"steps_s": TIMINGS["steps_s"], "avg_query_ms": avg, "num_queries": len(q["search"])}
    with open("reports/latency_report.json", "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(f"  Latency/query (ms): {avg} · steps (s): {TIMINGS['steps_s']}")


if __name__ == "__main__":
    start = time.time()
    search, reranker = build_pipeline()
    evaluate_pipeline(search, reranker)
    print(f"\nTotal: {time.time() - start:.1f}s")
