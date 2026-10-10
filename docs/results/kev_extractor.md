# Kết quả: extractor KevFormat so với r4

Trạng thái: **đã đăng ký trước, chưa chạy.** Phần "Giao thức" và "Cổng" được viết trước khi có bất kỳ con số dev nào của Kev, và không được sửa sau khi đã chạy. Kết quả điền vào các mục ở cuối.

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
| Loss | CE + head confidence + negative giả | chỉ CE (`loss_weighting: token`) |
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

_Chưa chạy._

## Kết quả test

_Chưa chạy._
