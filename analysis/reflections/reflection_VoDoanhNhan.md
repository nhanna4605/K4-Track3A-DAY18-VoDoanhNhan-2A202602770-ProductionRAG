# Individual Reflection — Lab 18: Production RAG

**Họ và tên:** Võ Doanh Nhân (MSSV 2A202602770)  
**Khóa:** K4 - Track 3A  
**Ngày hoàn thành:** 04/10/2026

---

## Phần 1: Mapping bài giảng (Lecture Mapping)

| Lecture Concept | Module | Hàm cụ thể | Observation & Phân tích |
|----------------|--------|-------------|--------------------------|
| Semantic chunking | M1 | `chunk_semantic()` | Chạy `compare_strategies` trên toàn kho: basic 51 chunk (TB 410 ký tự), semantic **208 chunk (TB 99 ký tự, min 6)**. Ngưỡng 0.85 với `all-MiniLM-L6-v2` (mô hình tiếng Anh) cắt quá vụn trên tiếng Việt, nên tôi không dùng semantic cho pipeline chính. |
| Hierarchical (parent–child) | M1 | `chunk_hierarchical()` | 11 cha (≤2048) và 99 con (≤256, TB 210). Bản `pipeline.py` mẫu chỉ đưa *đoạn con* vào LLM; tôi đo độ phủ số liệu của đáp án trong top-3: **0.71 (chỉ con) → 0.86 (trả về đoạn cha)**, cao hơn Naive (0.82). Đây là chỉ số tự đo trên 17/20 câu có số, không phải RAGAS. |
| Structure-aware | M1 | `chunk_structure_aware()` | 106 chunk theo tiêu đề Markdown, kèm `section` và `section_path` trong metadata; phù hợp kho quy chế vì mỗi mục (vd. "Phạt quá hạn") nằm trọn trong một chunk. |
| BM25 + Dense fusion | M2 | `segment_vietnamese()`, `BM25Search`, `DenseSearch`, `reciprocal_rank_fusion()` | BM25 chỉ khớp được cụm "nghỉ phép" sau khi đổi `_` thành khoảng trắng. RRF cộng `1/(k+rank+1)` nên không cần chuẩn hóa điểm cosine và điểm BM25. Truy hồi hybrid trung bình 248 ms/truy vấn. |
| Cross-encoder reranking | M3 | `CrossEncoderReranker.rerank()` | `bge-reranker-v2-m3` trên CPU mất **~8,9 s cho 20 đoạn** khi đo riêng (~11,5 s/truy vấn trong pipeline), vượt xa mục tiêu 150 ms. Context Precision tăng từ 0.925 lên 0.975 nhưng cái giá về độ trễ rất lớn nếu không có GPU. |
| RAGAS 4 metrics | M4 | `evaluate_ragas()`, `failure_analysis()` | Naive 0.800 / 0.719 / 0.925 / 0.925, Production 0.789 / 0.796 / 0.975 / 0.925 (Faithfulness / Relevancy / Precision / Recall). Hai lần chạy Production lệch tới ~0.017 ở Faithfulness nên chênh lệch nhỏ không đáng tin. Chỉ số thấp nhất ở các câu tệ nhất thường là Faithfulness. |
| Contextual embeddings / Enrichment | M5 | `_enrich_single_call()` (combined, 1 call/chunk), `contextual_prepend()` | 104 chunk làm giàu bằng 104 lần gọi (~424 s do gọi qua cổng API trung gian, ~4 s/lần). Câu ngữ cảnh đầu đoạn (vd. tên tài liệu và phiên bản) giúp phân biệt bản v1/v2 của cùng một chính sách. Không có API key thì fallback gắn "Trích từ <tên file>." vẫn chạy được. |

**Latency breakdown (từ `reports/latency_report.json`):**

| Bước | Thời gian |
|------|-----------|
| Chunking (M1) | 0.22 s |
| Enrichment (M5, 104 lần gọi LLM) | 424.3 s |
| Indexing BM25 + Dense (M2, bge-m3 trên CPU) | 60.5 s |
| Nạp reranker | ~0 s (nạp lười, trả giá ở truy vấn đầu) |
| Mỗi truy vấn: tìm kiếm / rerank / sinh câu trả lời | 247.6 ms / 11 513 ms / 2 360 ms |

---

## Phần 2: Khó khăn & Cách giải quyết (Challenges & Debugging)

- **Lỗi kỹ thuật gặp phải (Exact error message):**
  - `ERROR: ResolutionImpossible` khi `pip install -r requirements.txt`: *"langchain-community 0.2.19 depends on langsmith<0.2.0 and >=0.1.112"* — các ràng buộc `langchain-community<0.3` và `langchain-openai<0.2` xung đột với phần còn lại.
  - `openai.AuthenticationError: Error code: 401 - Incorrect API key provided: sk-***` khi gọi thẳng `api.openai.com`.
  - `Did not find openai_api_key, please add an environment variable OPENAI_API_KEY` ở bước RAGAS khi chưa có key.
  - `failed to connect to the docker API at npipe:////./pipe/dockerDesktopLinuxEngine` — Docker Desktop chưa chạy nên không có Qdrant server.
- **Nguyên nhân gốc rễ & Cách debug:**
  - Xung đột pip: tôi tạo venv riêng (không phá môi trường chung) và dùng một file requirements tạm nới ràng buộc hai gói langchain; `requirements.txt` của repo giữ nguyên.
  - Lỗi 401: key thực ra là của một cổng API trung gian tương thích OpenAI. Đặt `OPENAI_BASE_URL` và `OPENAI_API_BASE` trong `.env`, rồi kiểm tra riêng chat (`gpt-4o-mini`) và embeddings (`text-embedding-ada-002`) trước khi chạy cả pipeline, vì RAGAS dùng cả hai.
  - Thiếu key: RAGAS bọc trong `try/except` trả điểm 0.0 và in lý do thay vì bịa số; có key rồi mới chạy lại.
  - Docker: `DenseSearch` đã có sẵn nhánh dự phòng `QdrantClient(":memory:")` nên pipeline vẫn chạy được mà không cần container.
  - Chất lượng pipeline: bản đầu cho độ phủ số liệu 0.71, thấp hơn Naive 0.82. Soi từng câu cho thấy chỉ đoạn con 256 ký tự được đưa vào LLM nên mất ngữ cảnh. Sửa `pipeline.py` để tìm trên đoạn con nhưng gửi đoạn cha → 0.86.
- **Kiến thức còn thiếu & Cách khắc phục:**
  - Không biết RAGAS Faithfulness phạt các bước suy luận/tính toán. Đọc từng câu tệ nhất trong `ragas_details.json` mới thấy 4/5 câu có ngữ cảnh đúng nhưng lỗi nằm ở khâu sinh hoặc cách chấm, không phải truy hồi.
  - Điểm RAGAS dao động giữa các lần chạy (0.806 → 0.789 ở Faithfulness), nên tôi báo cáo cả hai lần và không khẳng định các chênh lệch nhỏ.

---

## Phần 3: Action Plan cho Project cá nhân (Application Plan)

> Lưu ý: tôi không biết đồ án cụ thể của học viên nên phần này viết cho một đồ án điển hình (chatbot hỏi đáp quy chế/tài liệu nội bộ tiếng Việt); cần thay bằng tên và hiện trạng đồ án thật khi áp dụng.

### Project: Chatbot hỏi đáp tài liệu nội bộ tiếng Việt

#### 1. Hiện trạng
- **Pipeline hiện tại:** Naive RAG — cắt theo đoạn văn, tìm kiếm vector đơn, lấy top-3 rồi cho LLM trả lời.
- **Vấn đề / Bottlenecks đang gặp:** Câu hỏi có từ khóa chính xác (số tiền, số ngày) hay trượt; câu hỏi xung đột phiên bản (chính sách cũ/mới) bị trả về bản cũ; LLM suy luận hoặc tính toán không bám tài liệu.

#### 2. Kế hoạch cải tiến
1. **Chunking strategy:** Structure-aware theo tiêu đề cho tài liệu quy chế, kết hợp parent–child (tìm trên con, trả cha) — đã đo cho độ phủ số liệu 0.86 so với 0.71 khi chỉ dùng chunk con. Không dùng semantic với mô hình embedding tiếng Anh.
2. **Search retrieval:** Hybrid BM25 (underthesea, đổi `_` thành khoảng trắng) + dense bge-m3 + RRF, vì BM25 bắt số liệu còn dense bắt ý nghĩa.
3. **Reranking:** Có, cross-encoder `bge-reranker-v2-m3` chỉ khi có GPU; trên CPU độ trễ ~11 s/truy vấn không chấp nhận được, khi đó dùng `FlashrankReranker` hoặc rerank top-10.
4. **Evaluation:** RAGAS 4 metrics trên bộ 20 câu cố định + lưu chi tiết từng câu (`ragas_details.json`) để chẩn đoán; chạy mỗi lần ≥2 lượt vì điểm dao động ~0.02.
5. **Enrichment:** Contextual prepend gộp một lần gọi (combined) để phân biệt phiên bản tài liệu; chi phí ~4 s/chunk nên chỉ chạy lại khi tài liệu đổi.

#### 3. Timeline triển khai
- **Tuần 1:** Dựng benchmark Naive + bộ câu hỏi chuẩn; đo baseline bằng RAGAS.
- **Tuần 2:** Thay chunking/hybrid search; sửa prompt sinh câu trả lời (cho suy luận có nêu căn cứ, `temperature=0`); đo lại.
- **Tuần 3:** Thêm rerank (đo độ trễ), truy hồi theo phiên bản tài liệu, và chạy lại RAGAS để so sánh.
