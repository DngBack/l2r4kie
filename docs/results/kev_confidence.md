# Kết quả: confidence trên extractor KevFormat so với r4

Trạng thái: **đã chạy trên `kev_r4` (2026-10-12). G4 trượt sát: review 30,4% > 29,9%; bắt lỗi 96,9% đạt. Ngang r4 trong khoảng tin cậy.**
- Các phần "Giao thức" và "Cổng G4" được viết trước khi có trace nào của `kev_r4`. Không sửa chúng sau khi đã chạy.
- Đã có hai thứ: kiểm tra phần port trên cache cũ của r4, và một probe tín hiệu trên adapter smoke (mục cuối, chỉ dùng để chọn tín hiệu đưa vào grid).

## Câu hỏi

Với extractor KevFormat (train chỉ bằng CE, như r4), head đọc trace của chính lượt decode có giảm được số field phải review xuống bằng hoặc thấp hơn r4 + attention không? Điều kiện là vẫn bắt ≥95% lỗi.

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

**Rủi ro đã biết trước:** cả r4 và Kev đều chỉ train bằng CE (`confidence_weight: 0`), nên không bên nào có lợi thế ở đây. Khác biệt nằm ở vị trí state: `h_end` của r4 ở sau `<|im_end|>` của cả câu trả lời JSON, còn `h_value` của Kev ở sau `<|box_end|>` của từng value. Probe dưới đây chạy trên adapter smoke, không phải `kev_r4`.

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
- **`h_value` mang thông tin tổng thể dù chỉ train bằng CE.** Một mình nó đạt 0,824, ngang summary token. Gộp với summary thì bổ sung cho nhau (0,868).
- **State ở layer giữa tốt hơn layer cuối.** `value@14` và `value@21` đạt 0,858, so với 0,824 ở layer cuối.
  - Giả thuyết: layer cuối (đã norm) chuyên cho việc đoán token tiếp theo sau `box_end`. Layer giữa còn giữ "đã đọc được gì".
  - Vì thế config chính cache layer 14 và 21.
- **`h_key` đoán được đúng/sai trước khi viết value** (0,814; 0,843 ở layer 21). Nó cho biết field khó hay dễ với tài liệu này, nên hợp làm prior hoặc làm query cho attention (mode `query`).
- `h_decide` không thêm gì so với `h_key`. Quyết định dừng một mình thì yếu.
- Probe không đo attention pooling trên token state, vì phần đó cần train. Đó là phần của grid.

## Kết quả trên `kev_r4`

Chạy 2026-10-12. Head đóng băng trước khi decode audit (`frozen_before_audit: true`).
File: `artifacts/kev-r4/confidence/selected/{selection,frozen,audit_report}.json`.

**Cohort đã cache** (24 field/tài liệu, budget 16.384, layer 14/21):

| Cohort | Tài liệu | Field | Có trace | Đúng |
|---|---:|---:|---:|---:|
| train | 796 | 14.895 | 14.761 | 12.848 (86,3%) |
| dev | 247 | 4.489 | 4.440 | 3.918 (87,3%) |
| calibration | 158 | 2.986 | 2.965 | 2.601 (87,1%) |
| risk_validation | 332 | 6.224 | 6.166 | 5.422 (87,1%) |
| audit | 347 | 6.580 | 6.526 | 5.675 (86,2%) |

**Head chính** (chọn trên dev theo luật đã đăng ký): `attention[value@14+value@21]`, tức attention pooling trên state value ở layer 14 và 21. Ngưỡng 0,9362 fit trên risk_validation; cả hai cận dưới 95% (Wilson 0,9500, bootstrap 0,9511) đều ≥ 95%.

### Cổng G4 (347 tài liệu audit, 6.580 field, 905 lỗi)

| | Kev | Ngưỡng | r4 trên cùng audit | |
|---|---:|---:|---:|---|
| G4a: review | **30,43%** [28,6–32,2] | ≤ 29,9% | 29,86% | **trượt** (+0,53 điểm) |
| G4b: bắt lỗi | **96,9%** [95,8–97,9] | ≥ 95% | 96,8% | đạt |

- **Không đạt G4.** Review vượt ngưỡng 0,53 điểm.
- **So ghép cặp với r4** (bootstrap theo tài liệu, 2.000 lần): Kev − r4 = **+0,56 điểm [−0,72; +1,82]**. Khoảng chứa 0, nên Kev và r4 ngang nhau. Kev có nhiều lỗi hơn để bắt (905 so với 852), vì EM thấp hơn một chút.
- Review lý thuyết (`oracle_review_rate`, bằng tỷ lệ lỗi): 13,1%. Head đang review gấp khoảng 2,3 lần mức tối thiểu.
- Không tính field bắt buộc review (54 field cắt cụt hoặc invalid): review 29,85%, bắt lỗi 96,7%.
- AUROC 0,966; ECE 0,012; AURC 0,0152; AUROC trong cùng (form, field) 0,875 (212 nhóm).
- Theo ngân sách review: 10% bắt 63,8% lỗi; 20% bắt 89,7%; 30% bắt 96,7%; 50% bắt 99,1%.

**Tách `fresh_audit` / `seen_audit`:**

| | Tài liệu | Review Kev | Review r4 | Kev − r4 [KTC 95%] | Bắt lỗi Kev / r4 |
|---|---:|---:|---:|---|---|
| fresh_audit | 145 | 30,7% | 29,0% | +1,7 [+0,1; +3,4] | 97,6% / 98,8% |
| seen_audit | 202 | 30,2% | 30,5% | −0,3 [−2,2; +1,5] | 96,4% / 95,5% |

Trên `fresh_audit`, Kev review nhiều hơn r4, và khoảng tin cậy vừa sát trên 0.

### Ablation theo family (cùng audit; ngưỡng mỗi family fit trên risk_validation)

| Family | Review | Bắt lỗi | AUROC | Kev − r4 [KTC 95%] |
|---|---:|---:|---:|---|
| mean[value] | 29,39% | 96,1% | 0,963 | −0,47 [−1,77; +0,81] |
| attention[value@21] | 29,51% | 97,2% | 0,969 | −0,35 [−1,67; +0,94] |
| **attention[value@14+value@21]** (chính) | 30,43% | 96,9% | 0,966 | +0,56 [−0,72; +1,82] |
| query[value] | 31,06% | 96,8% | 0,960 | |
| attention[value]+prior | 31,57% | 96,4% | 0,962 | |
| hybrid_mlp[value] | 31,60% | 97,0% | 0,961 | |
| attention[value] (head của r4, trên state Kev) | 31,66% | 96,4% | 0,962 | +1,79 [+0,41; +3,19] |
| attention[value+key] | 32,48% | 96,4% | 0,960 | |
| attention[value@14] | 34,10% | 97,7% | 0,964 | |
| attention[value+key+decide] | 36,16% | 97,4% | 0,958 | |
| hybrid[value] | 38,48% | 95,6% | 0,940 | |
| min log-prob (không train) | 45,24% | 96,2% | 0,923 | |
| mean log-prob (không train) | 57,45% | 97,9% | 0,881 | |
| end[value] | 63,12% | 96,1% | 0,879 | |

Hai family có review dưới 29,9% trên audit (`mean[value]`, `attention[value@21]`). Nhưng chúng không được chọn trên dev, và đọc chúng sau khi đã thấy audit là chọn lại trên tập test. Chúng chỉ là gợi ý cho lần đăng ký sau.

### Đọc kết quả

- **Câu hỏi đặt ra không được trả lời "có".** Trên extractor KevFormat, head đọc trace của chính lượt decode review 30,4%, ngang r4 + attention (29,9%). Khác biệt nằm trong nhiễu.
- **State ở layer giữa có giúp.** Cùng head attention, `value@14+value@21` review ít hơn `value` (layer cuối) 1,2 điểm. Khi đặt head của r4 lên state layer cuối của Kev, kết quả kém r4 rõ ràng: +1,8 [+0,4; +3,2].
- **`key` và `decide` không giúp.** Thêm vào làm review tăng, khớp với probe.
- **Thứ hạng family trên dev và audit lệch nhau khoảng ±1 điểm.** Khoảng cách giữa các head tốt nhất nhỏ hơn mức nhiễu khi chọn.

### Ghi chú vận hành (không ảnh hưởng kết quả)

Cache chạy trên GPU dùng chung (khoảng 20 giờ wall-clock). Trong lúc chạy đã sửa ba điểm trong decode và cache:
1. Fallback khi cuDNN attention lỗi: crop KV rồi chạy lại bước đó.
2. Thu gọn batch khi một nửa số nhánh đã đóng. Một value lặp tới 16.384 token từng giữ cả 24 nhánh trong batch, chậm khoảng 5 lần.
3. Checkpoint theo từng tài liệu trong mỗi part.

Cả ba đều giữ nguyên giá trị ở những bước vốn chạy được (có test). Riêng (2) đổi kích thước batch, nên có thể lệch ở mức bf16 trên các value rất dài. Các value đó phần lớn đã là sai hoặc cắt cụt.
