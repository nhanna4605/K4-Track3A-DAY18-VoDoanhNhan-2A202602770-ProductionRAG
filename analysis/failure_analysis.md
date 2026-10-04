# Failure Analysis — Lab 18: Production RAG

**Họ và tên học viên:** Võ Doanh Nhân (MSSV 2A202602770)  
**Khóa:** K4 - Track 3A  

Số liệu lấy từ `reports/naive_baseline_report.json` (Naive) và `reports/ragas_report.json` + `reports/ragas_details.json` (Production, chạy `python src/pipeline.py`). Điểm RAGAS có dao động giữa các lần chạy: lần chạy trước của cùng pipeline cho 0.806 / 0.797 / 0.975 / 0.925, nên chênh lệch dưới ~0.02 không đủ để kết luận.

---

## RAGAS Scores

| Metric | Naive Baseline | Production | Δ |
|--------|---------------|------------|---|
| Faithfulness | 0.8000 | 0.7889 | -0.0111 |
| Answer Relevancy | 0.7186 | 0.7961 | +0.0775 |
| Context Precision | 0.9250 | 0.9750 | +0.0500 |
| Context Recall | 0.9250 | 0.9250 | +0.0000 |

**Nhận xét thẳng:** Naive baseline đã rất mạnh vì kho chỉ có 26 văn bản nhỏ, 20 câu hỏi và chỉ lấy top-3. Production cải thiện rõ ở Answer Relevancy (+0.08) và Context Precision (+0.05); Context Recall không đổi; Faithfulness không cải thiện (nằm trong khoảng dao động giữa hai lần chạy). Hai PDF (`BCTC.pdf`, `Nghi_dinh_so_13-2023...pdf`) là bản scan ảnh nên không có text layer và bị bỏ qua ở cả hai pipeline.

## Bottom-5 Failures

Năm câu có điểm trung bình thấp nhất trong `reports/ragas_report.json` (mục `failures`). Ngữ cảnh/câu trả lời trích từ `reports/ragas_details.json`.

### #1
- **Question:** Nhân viên thử việc có được hưởng bảo hiểm sức khỏe PVI không?
- **Expected:** KHÔNG. Nhân viên thử việc chưa được hưởng gói PVI; chỉ có bảo hiểm xã hội bắt buộc.
- **Got:** "Không tìm thấy."
- **Worst metric:** faithfulness = 0.0 (answer_relevancy cũng 0.0; context_precision = 1.0, context_recall = 1.0)
- **Error Tree:** Output sai → Context đúng? **Có** (đã lấy cả `thu_viec.md` lẫn `bao_hiem_suc_khoe.md`, trong đó ghi gói PVI dành cho "nhân viên chính thức") → Query OK? Có → **LLM từ chối trả lời**.
- **Root cause:** Câu trả lời cần *suy luận* ("chỉ nhân viên chính thức" ⇒ nhân viên thử việc không thuộc diện), nhưng prompt sinh câu trả lời trong `pipeline.run_query` ra lệnh "CHỈ dựa trên context, nếu không có → Không tìm thấy", nên mô hình chọn từ chối thay vì kết luận từ hai đoạn đã có.
- **Suggested fix:** Sửa prompt ở khâu sinh (pipeline): cho phép kết luận logic từ ngữ cảnh và yêu cầu nêu căn cứ; chỉ nói "Không tìm thấy" khi ngữ cảnh thực sự thiếu. Không cần đụng tới M1/M2/M3.

### #2
- **Question:** Nhân viên tạm ứng 15 triệu, sau 20 ngày mới thanh toán. Bị phạt bao nhiêu?
- **Expected:** Hạn 15 ngày, quá hạn 5 ngày, phí 2%/tháng trên 15.000.000 = 300.000 VNĐ/tháng (≈ 50.000 VNĐ cho 5 ngày).
- **Got:** Nêu đúng luật (15 ngày, 2%/tháng) nhưng tính ra **5.000 VNĐ** (15.000.000 × 0,0667% × 5 — sai một bậc).
- **Worst metric:** faithfulness = 0.44 (context_recall = 0.5)
- **Error Tree:** Output sai → Context đúng? **Có** (đoạn "Thời hạn thanh toán" và "Phạt quá hạn" đều có trong ngữ cảnh duy nhất được trả về) → Query OK? Có → **LLM tính sai số học**.
- **Root cause:** Lỗi sinh câu trả lời: ngữ cảnh chỉ có công thức 2%/tháng, con số 5.000 do mô hình tự tính sai và không có trong tài liệu nên bị RAGAS coi là không trung thực. Ngoài ra top-3 chỉ trả về 1 đoạn cha (840 ký tự), nghĩa là bước rerank + gộp cha đã loại hết các đoạn khác.
- **Suggested fix:** Ở khâu sinh: yêu cầu trình bày từng bước tính và đặt `temperature=0`; với câu hỏi tính toán, gọi công cụ tính (calculator) thay vì để LLM tự nhẩm. Ở M3: tăng `RERANK_TOP_K` khi chỉ còn 1 đoạn cha sau khi gộp.

### #3
- **Question:** Có cần kích hoạt xác thực đa yếu tố (MFA) không?
- **Expected:** Có, theo chính sách mật khẩu v2.0 hiện hành; chính sách cũ v1.0 không yêu cầu MFA.
- **Got:** "Có, tất cả nhân viên bắt buộc kích hoạt MFA cho tài khoản email, VPN và các hệ thống nội bộ." (đúng, trích nguyên văn từ ngữ cảnh)
- **Worst metric:** context_recall = 0.5 (faithfulness = 0.67)
- **Error Tree:** Output sai? **Không — câu trả lời đúng.** → Context đúng? Một phần: có `mat_khau_v2.md` (có mục MFA) nhưng **thiếu `mat_khau_v1.md`** nên không có vế "chính sách cũ không yêu cầu MFA" của đáp án chuẩn → Query OK? Có.
- **Root cause:** Đây là câu hỏi **xung đột phiên bản**. Hệ thống chỉ lấy bản hiện hành nên thiếu nửa sau của đáp án chuẩn (context_recall 0.5). Điểm faithfulness 0.67 dù câu trả lời trích đúng nguyên văn cho thấy cả sự nhiễu của giám khảo RAGAS, không chỉ lỗi hệ thống.
- **Suggested fix:** Ở M2: với câu hỏi có từ "cần/hiện hành" mà tài liệu có nhiều phiên bản, truy hồi thêm bản cũ bằng bộ lọc metadata theo tên nguồn (`mat_khau_v1`/`v2`) để câu trả lời nêu được thay đổi giữa hai phiên bản.

### #4
- **Question:** Nhân viên được tài trợ khóa học 25 triệu, nghỉ việc sau 8 tháng hoàn thành khóa học. Phải hoàn trả bao nhiêu?
- **Expected:** Cam kết ≥ 1 năm; nghỉ sau 8 tháng là trước hạn → hoàn trả 100% = 25.000.000 VNĐ.
- **Got:** "Hoàn trả 100% chi phí đào tạo đã được tài trợ, tức là 25 triệu VNĐ." (đúng)
- **Worst metric:** faithfulness = 0.5 (precision = 1.0, recall = 1.0)
- **Error Tree:** Output đúng? **Có.** → Context đúng? Có (đoạn "Cam kết hoàn chi" nêu cả "ít nhất 1 năm" và "100% chi phí") → Query OK? Có → Lỗi nằm ở **cách chấm**, không phải ở pipeline.
- **Root cause:** Câu trả lời đúng nhưng chứa phép suy ra "8 tháng < 1 năm" và con số "25 triệu" lấy từ câu hỏi, không có nguyên văn trong ngữ cảnh; RAGAS tách thành các mệnh đề và chỉ chấp nhận mệnh đề suy ra trực tiếp từ ngữ cảnh. Đoạn thứ hai trong top-3 là `tam_ung.md` (không liên quan) — nhiễu nhỏ nhưng không ảnh hưởng đáp án.
- **Suggested fix:** Không cần sửa module. Nếu muốn nâng faithfulness: yêu cầu LLM nêu rõ căn cứ ("cam kết 1 năm, nghỉ sau 8 tháng nên phải hoàn 100%") để mỗi bước suy luận bám vào một câu của tài liệu.

### #5
- **Question:** Nghỉ phép không lương 20 ngày cần ai phê duyệt?
- **Expected:** 16–30 ngày cần CEO duyệt; lưu ý trên 14 ngày nhân viên tự đóng bảo hiểm.
- **Got:** "Cần phê duyệt của Giám đốc điều hành (CEO)." (đúng, nhưng thiếu lưu ý về bảo hiểm)
- **Worst metric:** faithfulness = 0.5 (precision = 1.0, recall = 1.0)
- **Error Tree:** Output đúng? Đúng một phần (thiếu ý bảo hiểm) → Context đúng? **Có** (đoạn "Nghỉ từ 16-30 ngày: CEO" và "Nghỉ trên 14 ngày, nhân viên cần tự đóng phần bảo hiểm" đều có) → Query OK? Có → mô hình **bỏ sót thông tin có sẵn** + suy luận "20 ngày ∈ [16, 30]" không nằm nguyên văn trong tài liệu.
- **Root cause:** Tương tự #4: bước suy luận số học trên khoảng giá trị bị RAGAS tính là mệnh đề không có căn cứ trực tiếp; prompt sinh cũng không yêu cầu trả lời đầy đủ các lưu ý liên quan.
- **Suggested fix:** Ở khâu sinh: thêm hướng dẫn "nêu điều kiện/lưu ý đi kèm trong ngữ cảnh". Ở đánh giá: đọc điểm faithfulness của nhóm câu suy luận số học với sự dè dặt vì giám khảo LLM nghiêm khắc.

## Case Study (cho presentation)

**Question chọn phân tích:** Nhân viên thử việc có được hưởng bảo hiểm sức khỏe PVI không? (câu tệ nhất: faithfulness 0.0, answer_relevancy 0.0)

**Error Tree walkthrough:**
1. Output đúng? → **Không.** Hệ thống trả "Không tìm thấy." trong khi đáp án là "KHÔNG, chưa được hưởng".
2. Context đúng? → **Có.** Ngữ cảnh gồm `thu_viec.md` và `bao_hiem_suc_khoe.md` (câu "gói bảo hiểm ... cho tất cả nhân viên chính thức"), context_precision = 1.0 và context_recall = 1.0.
3. Query rewrite OK? → Có, truy vấn không cần viết lại vì đã lấy đúng hai tài liệu.
4. Fix ở bước: **sinh câu trả lời (prompt trong `pipeline.run_query`)** — không phải chunking, search hay rerank. Điều này cho thấy điểm RAGAS thấp không phải lúc nào cũng do truy hồi: ở bốn trong năm câu tệ nhất, ngữ cảnh đã đủ và lỗi nằm ở khâu sinh/chấm.

**Nếu có thêm 1 giờ, sẽ optimize:**
- Viết lại prompt sinh câu trả lời (cho phép suy luận có nêu căn cứ, đặt `temperature=0`, bắt buộc liệt kê các bước tính) rồi chạy lại để xem faithfulness có vượt 0.85 không.
- Thêm truy hồi theo phiên bản (v1/v2) cho câu hỏi xung đột phiên bản như câu MFA.
- Giảm độ trễ rerank (hiện ~11,5 giây/truy vấn trên CPU) bằng `FlashrankReranker` hoặc chỉ rerank top-10 thay vì top-20.
