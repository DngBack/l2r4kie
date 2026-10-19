# Luật dạng value: review bắt buộc khi value có dạng field chưa từng có

Trạng thái: **đã triển khai, đã fit lại ngưỡng trên risk_validation (2026-10-19); chưa có audit sạch.** Bundle mới: `artifacts/kev-r4/confidence/selected-shapes/`. Bundle cũ `selected/` giữ nguyên.

## Ý tưởng

Head chính bỏ sót một nhóm lỗi tự tin: thiếu hoặc thừa chữ số, đọc nhầm dấu phân cách (`10/01/2023` thành `10 /01 /2023`, `35°` thành `35º`). Model viết các value này rất chắc tay, nên tín hiệu bất định trong trace không bắt được.

Nhiều field có định dạng ổn định: ngày, mã, số tiền. Nếu value dự đoán có dạng mà field đó chưa từng có trong nhãn train, thì gần như chắc chắn sai.

- **Dạng** (`shape`): chữ số thành `9`, chữ cái thành `a`, ký tự khác giữ nguyên. Ví dụ `12/03/2024` thành `99/99/9999`.
- **Học luật** (`ShapeRules.fit`): chỉ dùng **nhãn tập train của extractor**, không dùng cohort confidence nào.
  - Chỉ xét field scalar có value chứa chữ số.
  - Giữ các dạng chiếm ≥ 2% số value của field.
  - Field có luật khi có ≥ 20 value, giữ không quá 5 dạng, và các dạng giữ lại phủ ≥ 95% số value.
- **Áp luật** (`ShapeRules.violates`): field `ok`, scalar, value chứa chữ số và có dạng ngoài tập cho phép thì review bắt buộc, lý do `field_shape_violation`. Head không có quyền chấp nhận field này, dù confidence cao đến đâu.
- **Khóa theo `field_id`, không theo (form, field)**, vì request lúc infer không có form. Trên hai cohort, hai cách khóa gắn cờ gần như cùng các field:

| Khóa | Số luật | Cờ dev (sai) | Cờ risk_validation (sai) |
|---|---:|---:|---:|
| (form, field_id) | 740 | 67 (67) | 107 (105) |
| `field_id` | 528 | 62 (62) | 99 (98) |

## Triển khai

| Chỗ | Thay đổi |
|---|---|
| `confidence/shapes.py` | `shape`, `ShapeRules` (fit, violates, flag_rows, JSON) |
| `confidence/policy.py` | `is_mandatory` / `needs_review` / `review_reason` nhận `shape_violation`; `_usable` loại row có cờ; `queue_fields` đọc `shape_violation` |
| `eval/review_metrics.py` | `evaluate_policy`, `oracle_review_rate`, `review_curve`, `budgets`, `bootstrap_comparison` coi row có cờ là review bắt buộc |
| `pipelines/finalize.py` | `extractor_shape_rules(config)`; gắn cờ dev và risk_validation **trước khi** fit ngưỡng bảo thủ; ghi `shape_rules.json` vào bundle (nằm trong fingerprint); `bundle.json` có `shape_rules: true` |
| `confidence/bundle.py`, `pipelines/infer.py` | bundle đọc luật; `review(results, scores, requests)` áp luật lúc serve. Nếu `bundle.json` ghi có luật mà thiếu `shape_rules.json` thì load báo lỗi, vì ngưỡng fit kèm luật sẽ chấp nhận đúng các value mà luật lẽ ra phải chặn |
| `pipelines/audit.py` | áp luật của từng bundle lên các row audit |
| config | `shape_rules: true` (mặc định `false`) |

Calibration không đổi, vì nó chỉ map logit sang xác suất. Head và calibration của `selected-shapes` giống `selected` (chỉ lệch khoảng 1e-8 do nhiễu số thực khi chạy đa luồng trên CPU). Chỉ ngưỡng thay đổi.

```bash
# selection.json, trials.json và head.pt/head_config.json của mỗi family được chép từ selected/
l2r4kie finalize --config configs/confidence/kev.yaml \
    --set output=artifacts/kev-r4/confidence/selected-shapes --set shape_rules=true
```

Test: `tests/test_shapes.py`, `tests/test_policy.py` (hai test mới), `tests/test_pipelines.py`. Test serving parity giờ chạy cả khi có và không có luật: confidence, quyết định review và lý do `field_shape_violation` lúc infer khớp với audit.

## Kết quả (head chính `attention[value@14+value@21]`, mục tiêu bắt 95% lỗi)

Ngưỡng bảo thủ (cả cận dưới Wilson lẫn cận bootstrap theo tài liệu phải ≥ 95%): **0,9362 xuống 0,9220**.

| Cohort | Cờ (sai / tổng) | Review: không luật → có luật | Bắt lỗi | Mức sàn oracle |
|---|---:|---:|---:|---:|
| risk_validation (332 tài liệu, 802 lỗi) | 98 / 99 (1,6% số field) | **29,18% → 27,23%** (−1,95) | 96,51% → 96,51% | 12,2% |
| dev (247 tài liệu, 571 lỗi) | 62 / 62 (1,4%) | **28,25% → 26,13%** (−2,12) | 96,85% → 97,02% | 12,1% |

Review cần để đạt từng mức bắt lỗi (ngưỡng thấp nhất trên chính cohort đó, không bảo thủ):

| Bắt lỗi | risk_validation | dev |
|---|---|---|
| 95% | 26,5% → 24,7% | 24,3% → 22,9% |
| 97% | 30,1% → 27,7% | 29,0% → 26,1% |
| 98% | 34,5% → 31,2% | 34,7% → 31,3% |
| 99% | 42,8% → **36,5%** | 41,2% → **38,0%** |

Mức bắt lỗi mục tiêu càng cao thì luật càng có lợi, vì các lỗi nó bắt chính là lỗi có confidence cao nhất. Luật không tốn GPU: chỉ là so khớp chuỗi trên value đã decode.

## Giới hạn

- **Chưa có số đo sạch.** Ý tưởng và tham số của luật (2%, 95%, 5 dạng, 20 mẫu) được đặt ra khi phân tích lỗi bỏ sót trên risk_validation, rồi kiểm lại trên dev. Cả hai cohort đều đã thấy, nên mức giảm khoảng 2 điểm có thể hơi lạc quan. Audit cũ (347 tài liệu) đã dùng để quyết định G4 nên **không chạy lại**. Để có số sạch cần một cohort audit mới (xem ghi chú cuối `agreement_signal.md`: còn khoảng 1.090 tài liệu, nhưng chỉ thuộc 16/58 form).
- **Review theo tài liệu vẫn khoảng 99–100%.** Với 24 field mỗi tài liệu, gần như tài liệu nào cũng có ít nhất một field cần xem. Luật giảm số field phải xem, nhưng không bớt được số tài liệu.
- Luật chỉ có cho field có định dạng ổn định và có chữ số trong nhãn train (528 field id). Field mới hoặc field văn bản tự do không có luật.
- Nếu tập train của extractor đổi thì phải học lại luật. `finalize` học luật từ config extractor mỗi lần chạy.
