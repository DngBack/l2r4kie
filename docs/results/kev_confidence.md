# Kết quả: confidence trên extractor KevFormat so với r4

Trạng thái: **đã đăng ký trước, chưa chạy trên `kev_r4`.**
- Các phần "Giao thức" và "Cổng G4" được viết trước khi có trace nào của `kev_r4`. Không sửa chúng sau khi đã chạy.
- Đã có hai thứ: kiểm tra phần port trên cache cũ của r4, và một probe tín hiệu trên adapter smoke (mục cuối, chỉ dùng để chọn tín hiệu đưa vào grid).

## Câu hỏi

Với extractor KevFormat (train chỉ bằng CE, không có head BCE hay negative giả), head đọc trace của chính lượt decode có giảm được số field phải review xuống bằng hoặc thấp hơn r4 + attention không? Điều kiện là vẫn bắt ≥95% lỗi.

## Tín hiệu dùng được (tại mỗi cặp key–value)

| Tín hiệu | Vị trí trong chuỗi | Ý nghĩa | Ở r4 |
|---|---|---|---|
| `value` (`h_value`) | state sau khi đưa `<|box_end|>` vào | đã đọc xong toàn bộ value: thông tin tổng thể | `h_end`, sau `<|im_end|>` |
| `key` (`h_key`) | `<|object_ref_end|>` | đã đọc xong tên field, chưa viết value | không có |
| `decide` (`h_decide`) | `<|box_start|>` | ngay trước khi viết value | không có |
| `value@L`, `key@L`, `decide@L` | 3 state trên ở `hidden_states[L]` | các layer giữa | không có |
| token state | state sau mỗi token value (≤ 512 token mỗi field, ưu tiên token có log-prob thấp) | cho attention / mean pooling | có (≤ 256 token) |
| thống kê token | log-prob, entropy chuẩn hóa, margin, prob, **log-prob của close** | 5 cột cho mỗi token | 4 cột |
| `close` | thống kê của quyết định dừng | | không có |
| summary | 27 cột: thống kê token, độ dài, tỷ lệ chữ số, loại value | | 21 cột |

Tất cả đều lấy từ lượt decode tạo ra value. Không có lượt encode ảnh hay verifier thứ hai. Phần serving dùng chung code với phần cache, nên điểm khi serving trùng với điểm lúc audit (`tests/test_pipelines.py`).

## Giao thức

**Hai hệ thống**

| | r4 (baseline) | Kev (ứng viên) |
|---|---|---|
| Extractor | `r4_2m_12f` | `artifacts/kev-r4` (`configs/extractor/kev_r4.yaml`) |
| Budget decode | 256 token | 16.384 token (như khi serving) |
| Head đã chọn | attention, seed 109, ngưỡng 0,9597 | chọn trên dev theo grid dưới đây |
| Prediction audit | `token-review-v2/selected/attention/audit_predictions.jsonl` (repo cũ, chỉ đọc) | `artifacts/kev-r4/confidence/selected/heads/<family>/audit_predictions.jsonl` |

**Cohort:** dùng lại nguyên kế hoạch v2 của r4 (`token-review-v2/cache/cohort_plan.json`).

| Cohort | Tài liệu | Dùng để |
|---|---:|---|
| train | 796 | train head |
| dev | 247 | chọn head |
| calibration | 158 | chọn và fit calibration |
| risk_validation | 332 | chọn ngưỡng (cận dưới Wilson và bootstrap theo tài liệu đều ≥ 95%) |
| audit | 347 = `fresh_audit` 145 + `seen_audit` 202 | đo một lần, sau khi đóng băng |

- Đã kiểm tra: không tài liệu nào của 5 cohort nằm trong tập train của `kev_r4` (holdout 15%, seed 42). `cache-traces` tự kiểm tra lại điều này trước khi load model.
- Mỗi tài liệu lấy 24 field đầu, như r4. Đúng điều kiện ghép cặp với r4 trên cùng (tài liệu, field).
- `seen_audit` đã được người xem ở lượt v1 của r4 (không phải extractor đã thấy). Phần này báo cáo riêng và không dùng để chọn gì.

**Grid** (`configs/confidence/kev.yaml`, 68 tổ hợp × 3 seed (107, 108, 109) × 3 snapshot (75, 150, 300 step)):
1. Grid v2 của r4, giữ nguyên: `end`, `hybrid`, `hybrid_mlp`, `mean`, `attention` trên `[value]`, L2 ∈ {0,001; 0,01}, ranking ∈ {0; 0,3}.
2. `attention` + prior theo field.
3. `hybrid` và `attention` trên `[value, key]`, `[value, key, decide]`, `[value@14]`, `[value@21]`, `[value@14, value@21]`.
4. `query`: attention pooling với query là `h_key`.
5. Hai heuristic không train (min / mean log-prob) cũng tham gia so sánh.

Mỗi family giữ snapshot tốt nhất trên dev. Head chính là family có review thấp nhất ở mức bắt 95% lỗi trên dev, hòa thì xét AURC, rồi AUROC.

**Lệnh**
```bash
C=configs/confidence/kev.yaml
uv run l2r4kie cache-traces --config $C --split train dev calibration risk_validation
uv run l2r4kie select-heads --config $C
uv run l2r4kie finalize --config $C
uv run l2r4kie cache-traces --config $C --split audit          # chỉ sau finalize
uv run l2r4kie audit --config $C --baseline-predictions \
  /home/jovyan/bachdx2/l2r4kie/artifacts/token-review-v2/selected/attention/audit_predictions.jsonl
```

## Cổng G4 (đo trên toàn bộ 347 tài liệu audit, head chính)

| | Điều kiện | Mốc r4 |
|---|---|---|
| G4a | review ≤ **29,9%** | 29,9% [28,1–31,7] |
| G4b | bắt lỗi ≥ **95%** (điểm ước lượng) | 96,8% [95,6–98,0] |

Báo cáo kèm, không phải cổng:
- Khoảng 95% bootstrap theo tài liệu.
- Tách `fresh_audit` và `seen_audit`.
- So sánh ghép cặp với r4 trên cùng field (mỗi hệ thống tính với value và lỗi của chính nó): chênh lệch review và khoảng 95%.
- `head_error_recall` (không tính field bắt buộc review).
- AUROC trong cùng field.
- Review lý thuyết (`oracle_review_rate`) của mỗi extractor.
- Ablation theo family: `value` / `+key` / `+key+decide` / `value@L` / `query` / prior.

**Rủi ro đã biết trước:** `h_end` của r4 được train cùng một head BCE và negative giả. `h_value` của Kev chỉ qua CE, nên có thể chứa ít thông tin về đúng/sai hơn. Probe dưới đây cho thấy nó vẫn tách được đúng/sai, nhưng probe chạy trên một adapter khác.

## Đã kiểm tra

**1. Calibration và policy chuyển đúng.**
- Lệnh: `l2r4kie finalize --legacy-selection .../token-review-v2/selected --cache .../token-review-v2/cache`. Lệnh này fit lại calibration và ngưỡng cho 5 head cũ trên cache cũ.

| Head | Ngưỡng mới | Ngưỡng cũ | Chênh | Review trên risk |
|---|---:|---:|---:|---:|
| attention | 0,9597253951 | 0,9597253875 | 7,6e-9 | 29,96% |
| mean | 0,9612490104 | 0,9612490170 | −6,6e-9 | 31,94% |
| hybrid_mlp | 0,9641090598 | 0,9641090531 | 6,7e-9 | 34,35% |
| hybrid | 0,9510181607 | 0,9510181607 | −3,1e-11 | 49,41% |
| end | 0,9604210487 | 0,9604044587 | 1,7e-5 | 69,23% |

- Chênh lệch chỉ ở mức sai số làm tròn float32/float64.
- `end` lệch lớn hơn một chút, nhưng vẫn rất nhỏ. Lý do: head tuyến tính trên 1.536 chiều, nên nhạy hơn với thứ tự cộng.

**2. Probe tín hiệu trên adapter smoke** (`scripts/probe_signals.py`, `artifacts/kev-smoke/probe/probe_report.json`)

Cohort probe:
- 116 tài liệu: 1 tài liệu/form từ train_reserve và 1 tài liệu/form từ dev. Không tài liệu nào thuộc gradient của extractor.
- 24 field/tài liệu, budget 256, layer 7/14/21.

Kết quả: 2.138 field, trong đó 2.038 field decode xong. Adapter smoke còn yếu:
- **63% value sai.**
- Review lý thuyết 61,9%.

Vì vậy review ở mức 95% gần như bị chặn sẵn, và **chỉ AUROC có ý nghĩa để xếp hạng tín hiệu.**

Cách probe:
- Hồi quy logistic L2 trên từng nhóm tín hiệu.
- CV 5 fold, chia theo tài liệu.
- Hệ số phạt chọn trên chính CV đó. Cách này lạc quan như nhau cho mọi nhóm.

| Tín hiệu | Chiều | AUROC |
|---|---:|---:|
| close (quyết định dừng) | 5 | 0,654 |
| decide@7 / decide@14 | 1.536 | 0,803 / 0,806 |
| decide (layer cuối) | 1.536 | 0,811 |
| value@7 | 1.536 | 0,813 |
| key (layer cuối) / key@14 | 1.536 | 0,814 / 0,814 |
| key@7 | 1.536 | 0,821 |
| **value (layer cuối, `h_value`)** | 1.536 | **0,824** |
| summary (thống kê token) | 27 | 0,829 |
| decide@21 | 1.536 | 0,841 |
| key@21 | 1.536 | 0,843 |
| **value@21** | 1.536 | **0,858** |
| **value@14** | 1.536 | **0,858** |
| value + summary | 1.563 | 0,868 |
| value@7 + summary | 1.563 | 0,875 |
| key + decide + value + summary | 4.635 | 0,875 |
| value@21 + summary | 1.563 | 0,885 |
| **value@14 + summary** | 1.563 | **0,892** |

Nhận xét (sơ bộ, adapter yếu, chỉ probe tuyến tính):
- **`h_value` vẫn mang thông tin tổng thể dù không train cùng BCE.** Một mình nó đạt 0,824, ngang summary token. Gộp với summary thì bổ sung cho nhau (0,868).
- **State ở layer giữa tốt hơn layer cuối.** `value@14` và `value@21` đạt 0,858, so với 0,824 ở layer cuối.
  - Giả thuyết: layer cuối (đã norm) chuyên cho việc đoán token tiếp theo sau `box_end`. Layer giữa còn giữ "đã đọc được gì".
  - Vì thế config chính cache layer 14 và 21.
- **`h_key` đoán được đúng/sai trước khi viết value** (0,814; 0,843 ở layer 21). Nó cho biết field khó hay dễ với tài liệu này, nên hợp làm prior hoặc làm query cho attention (mode `query`).
- `h_decide` không thêm gì so với `h_key`. Quyết định dừng một mình thì yếu.
- Probe không đo attention pooling trên token state, vì phần đó cần train. Đó là phần của grid.

## Kết quả trên `kev_r4`

*(chưa chạy: điền sau `audit`)*
