# Kết quả: extractor KevFormat so với r4

Trạng thái: **đã chạy: dev đạt G1–G3; test đã chạy (2026-10-12).** Phần "Giao thức" và "Cổng" được viết trước khi có bất kỳ con số dev nào của Kev, và không được sửa sau khi đã chạy. Kết quả điền vào các mục ở cuối.

## Câu hỏi

Đổi sang KevFormat (value viết dạng text, đặt giữa marker `<|box_start|>`/`<|box_end|>`, không giới hạn 256 token) có giữ được độ chính xác của r4 trên field scalar không, khi train cùng một công thức?

## Hai hệ thống

| | r4 (baseline) | Kev (ứng viên) |
|---|---|---|
| Train | `configs/experiments/r4_2m_12f.yaml` (repo cũ) | `configs/extractor/kev_r4.yaml` |
| Model, LoRA | Qwen2-VL-2B, r16/α32 q/k/v/o | như r4 |
| Ảnh, field/tài liệu, step, lr, seed | 2,1 MP, 12, 1.000, 1e-4, 42 | như r4 |
| Value | JSON | text, đóng bằng `box_end` |
| Budget decode khi eval | 256 token | 16.384 token |
| Loss | chỉ CE (`confidence_weight: 0`, `ranking_weight: 0`) | chỉ CE (`loss_weighting: token`) |
| Prediction | `.../training-optimization/r4_2m_12f/{dev,test}.jsonl` (repo cũ, chỉ đọc) | `artifacts/kev-r4/eval-{dev,test}/predictions.jsonl` |

## Giao thức

**Tập eval** (giống hệt r4; đã kiểm tra rằng `evaluation_documents` cho ra đúng các cặp (tài liệu, field) mà r4 đã chấm):

| Split | Tài liệu | Form | Field | Cách chọn |
|---|---|---|---|---|
| dev | 116 | 58 | 2.138 (2.038 scalar, 100 mảng) | Danh sách trong `cohort_plan.json` của r4 (2 tài liệu/form), 24 field đầu mỗi tài liệu |
| test | 116 | 47 | 2.150 (2.052 scalar, 98 mảng) | 116 tài liệu đầu của split test (selection seed 42), 24 field đầu |

- Ở dev, thứ tự tài liệu khác với file của r4. Tập cặp thì giống hệt. Mọi metric và `compare` đều không phụ thuộc thứ tự.
- Mọi field được hỏi đều được decode và chấm, kể cả mảng dài. Không dùng ground truth để lọc field hay chọn budget.
- `truncated` và `invalid_*` đều tính là sai.

**Comparator: `text`.**
- Chuẩn hóa NFC và bỏ khoảng trắng hai đầu. Phân biệt hoa/thường, dấu câu và số 0 ở đầu.
- Scalar được so theo dạng text, ví dụ `"123"` bằng `123`. Lý do: Kev viết value dạng text, nên không thể sai kiểu JSON.
- Mảng phải là mảng và phải khớp từng phần tử, theo đúng thứ tự.
- Với prediction r4, `text` và `json` cho ra cùng EM, trên cả dev lẫn test.

**Lệnh** (chạy từ gốc repo):
```bash
R4=/home/jovyan/bachdx2/l2r4kie/artifacts/training-optimization
# dev
uv run l2r4kie evaluate --config configs/extractor/kev_r4.yaml --adapter artifacts/kev-r4 \
  --split dev --documents-file $R4/cohort_plan.json --fields 24 --output artifacts/kev-r4/eval-dev
uv run l2r4kie compare --a $R4/r4_2m_12f/dev.jsonl --b artifacts/kev-r4/eval-dev/predictions.jsonl --kind scalar
uv run l2r4kie compare --a $R4/r4_2m_12f/dev.jsonl --b artifacts/kev-r4/eval-dev/predictions.jsonl
# test: chỉ chạy khi G1-G3 đều đạt trên dev
uv run l2r4kie evaluate --config configs/extractor/kev_r4.yaml --adapter artifacts/kev-r4 \
  --split test --limit 116 --fields 24 --output artifacts/kev-r4/eval-test
```
Budget decode lấy từ `format.max_value_tokens` (16.384). Có thể thêm `--set device=cuda:1`, vì device không ảnh hưởng đến provenance.

## Baseline r4 (chấm lại bằng code mới)

Tái lập số đã báo cáo: chấm lại bằng `json` cho EM dev 86,997% và test 88,977%, khớp `TRAINING_OPTIMIZATION_REPORT.md`. Cờ `correct` lưu trong file khớp 100%.

| Metric (comparator `text`) | dev | test |
|---|---|---|
| EM (tất cả field) | 86,997% | 88,977% |
| **EM scalar** | **90,334%** (2.038) | 92,203% (2.052) |
| EM mảng | 19,00% (100) | 21,43% (98) |
| EM macro theo form | 85,94% | 87,34% |
| Tỷ lệ tài liệu đúng hết | 12,07% | 7,76% |
| **Tỷ lệ output tọa độ** | **0%** | 0% |
| **Tỷ lệ cắt cụt** | **2,292%** (49/2.138; cả 49 đều là mảng) | 2,186% (47/2.150) |
| Status | ok 2.086, truncated 49, invalid_json 3 | ok 2.097, truncated 47, invalid_json 6 |
| Mảng: F1 theo dòng / tỷ lệ ô đúng | 14,7% / 7,6% | 9,6% / 4,9% |

Lỗi trên dev (sau đánh giá, chỉ để phân tích): other_content 111, truncated 49, diacritics_or_case 49, array_content 28, missing_value 14, spurious_nonempty 12, digit_substitution 7, invalid_json 3, case_only 3, wrong_json_type 2.

## Cổng (dev)

| Cổng | Điều kiện | Ngưỡng |
|---|---|---|
| G1 | EM scalar của Kev ≥ EM scalar của r4 − 1 điểm % | **≥ 89,334%** |
| G2 | Tỷ lệ output tọa độ của Kev (trên mọi field) | **< 0,5%** |
| G3 | Tỷ lệ cắt cụt của Kev (trên mọi field) ≤ của r4 | **≤ 2,292%** |

- **Cách quyết định:** G1 xét trên số điểm, không xét khoảng tin cậy. `compare --kind scalar` (bootstrap theo tài liệu, 10.000 lần, seed 42) được báo cáo kèm để thấy độ bất định, nhưng không đổi kết luận.
- **Đọc G3:** hai hệ thống có budget khác nhau. Ở r4, cắt cụt nghĩa là "mảng dài hơn 256 token". Ở Kev, cắt cụt nghĩa là "16.384 token mà vẫn không đóng value", tức là model bị lặp. G3 vẫn được giữ đúng như kế hoạch. Tỷ lệ cắt cụt riêng trên field scalar của r4 là 0%; số này được báo cáo kèm, không phải cổng.
- **Nếu đạt cả ba:** chạy test một lần, ghi kết quả, rồi sang step 6.
- **Nếu không đạt:** không chạy test. Dừng lại để quyết định: chạy phương án dự phòng (`format.close=im_end`, hoặc `loss_weighting: field` nếu G1 trượt vì mảng dài lấn át scalar), hay giữ format cũ.

## Kết quả dev

Chạy ngày 2026-10-11, `artifacts/kev-r4/eval-dev/metrics.json` (116 tài liệu, 2.138 field). **Đạt cả ba cổng nên được chạy test.**

| Cổng | Kev | Ngưỡng | |
|---|---:|---:|---|
| G1: EM scalar | **89,794%** (1.830/2.038) | ≥ 89,334% | đạt (thấp hơn r4 0,54 điểm, biên cho phép là 1) |
| G2: output tọa độ | **0%** | < 0,5% | đạt |
| G3: cắt cụt | **0,281%** (6/2.138; cả 6 đều là mảng) | ≤ 2,292% | đạt |

| Metric (comparator `text`) | r4 | Kev | Kev − r4 [KTC 95%, bootstrap theo tài liệu] |
|---|---:|---:|---|
| EM scalar | 90,334% | 89,794% | −0,54 [−1,90; +0,79] |
| EM mảng | 19,00% | 22,00% | +3,0 [−2,9; +9,6] |
| EM (tất cả field) | 86,997% | 86,623% | −0,37 [−1,64; +0,90] |
| EM macro theo form | 85,94% | 85,12% | |
| Tỷ lệ tài liệu đúng hết | 12,07% | 11,21% | |
| Mảng: F1 theo dòng / tỷ lệ ô đúng | 14,7% / 7,6% | 21,8% / 29,5% | |
| Status | ok 2.086, truncated 49, invalid_json 3 | ok 2.120, truncated 6, invalid_array 12 | |

- Field scalar chỉ một bên sai: Kev sai mà r4 đúng 94, r4 sai mà Kev đúng 83. Hai hệ thống ngang nhau trong phạm vi nhiễu: khoảng tin cậy chứa 0 ở mọi dòng.
- Kev gần như hết cắt cụt (49 → 6) nhờ budget dài. 12 mảng `invalid_array` là các value không parse được thành mảng.
- Lỗi trên dev (chỉ để phân tích): other_content 116, array_content 60, diacritics_or_case 59, spurious_nonempty 13, invalid_array 12, missing_value 9, truncated 6, digit_substitution 5, internal_whitespace 4, case_only 2.
  - Lỗi dấu/hoa thường nhiều hơn r4 (59 so với 49). Đây là điểm yếu duy nhất thấy rõ.
  - `array_content` tăng (28 → 60) vì các mảng trước đây bị cắt cụt giờ được viết hết, và sai ở nội dung.

## Kết quả test

Chạy một lần sau khi dev đạt G1–G3. File: `artifacts/kev-r4/eval-test/metrics.json` (116 tài liệu, 47 form, 2.150 field).

| Metric (comparator `text`) | r4 | Kev | Kev − r4 [KTC 95%, bootstrap theo tài liệu] |
|---|---:|---:|---|
| EM scalar | 92,203% | 91,374% | −0,83 [−2,09; +0,35] |
| EM mảng | 21,43% | 22,45% | +1,0 [−2,1; +4,5] |
| EM (tất cả field) | 88,977% | 88,233% | −0,74 [−1,94; +0,38] |
| EM macro theo form | 87,34% | 86,03% | |
| Tỷ lệ tài liệu đúng hết | 7,76% | 10,34% | |
| Output tọa độ | 0% | 0% | |
| Cắt cụt | 2,186% | 0,279% (6/2.150) | |
| Status | ok 2.097, truncated 47, invalid_json 6 | ok 2.137, truncated 6, invalid_array 7 | |

- Giống dev: Kev và r4 ngang nhau (khoảng tin cậy chứa 0). Điểm EM scalar của Kev thấp hơn khoảng 0,8 điểm, vẫn trong biên 1 điểm của G1.
- Cắt cụt giảm khoảng 8 lần. Lỗi dấu và hoa/thường vẫn là loại lỗi lớn thứ ba (56).
- Chi phí: các field lặp tới budget 16.384 token làm latency có đuôi dài. Trên dev, trung vị 10,7 s/tài liệu nhưng 6 tài liệu mất 730–2.240 s, trên GPU dùng chung, trước khi có bản sửa thu gọn batch. Xem `kev_confidence.md`, mục "Ghi chú vận hành".
