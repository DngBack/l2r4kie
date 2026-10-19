# Audit mới (audit2): đăng ký trước khi decode

Trạng thái: **đăng ký 2026-10-19, trước khi decode bất kỳ tài liệu audit2 nào.** Phần "Kết quả" chỉ được điền sau khi chạy `audit`. Cổng và biên dưới đây không được sửa sau khi đã thấy số.

## Vì sao cần

- Luật dạng (`shape_rules.md`) được thiết kế khi xem lỗi trên risk_validation và dev, nên số trên hai cohort đó có thể lạc quan.
- Merge LoRA và fast decode (`infer_speed.md`) đổi numerics. Value, trace và ngưỡng đều được fit lại trên cache mới, nhưng cần một lần đo độc lập để biết có nên chuyển serving hay không.
- Audit cũ (347 tài liệu) đã dùng cho G4, nên không dùng lại.

## Cohort

`artifacts/kev-r4/confidence/audit2/cohort_plan.json`, tạo bằng `scripts/plan_fresh_audit.py` (seed 211, tối đa 20 tài liệu mỗi form).

- Cohort train, dev, calibration, risk_validation giữ nguyên của plan v2, vì các head đã fit trên chúng và `audit` kiểm tra không trùng.
- `audit` mới gồm **318 tài liệu, 16 form** (20 tài liệu mỗi form; riêng `phieu-can-thiep-te-nga` có 18), khoảng 6.286 field (24 field đầu mỗi tài liệu).
- Đã loại:
  - mọi cohort của plan v2, gồm 347 tài liệu audit cũ;
  - 1.997 tài liệu trong danh sách loại của r4 (danh sách này đã chứa cả 232 tài liệu eval của extractor, `eval-dev` và `eval-test`);
  - tập train của extractor.

  Pool còn lại 1.092 tài liệu.
- **Giới hạn:** chỉ 16/58 form còn tài liệu chưa dùng. Audit2 nói về các form này, không thay G4.

## Hệ thống so sánh

Head chính `attention[value@14+value@21]`. Mỗi hệ thống dùng value, lỗi và ngưỡng đã đóng băng của chính nó.

| Tên | Bundle | Decode | Cache audit2 |
|---|---|---|---|
| **S** (đang serve) | `selected-shapes` | LoRA chưa merge, eager | `cache-audit2` (train/dev/calibration/risk_validation trỏ tới `cache/`) |
| **M** | `selected-merged` | `merge_adapter` | `cache-merged` |
| **F** | `selected-fast` | `merge_adapter` + `fast_decode` | `cache-fast` |
| R (tham chiếu) | `selected` (không luật, ngưỡng 0,9362) | như S | dùng lại trace của S |

M và F giữ nguyên head đã chọn. Chỉ calibration, ngưỡng và luật dạng được fit lại (`finalize`) trên cache của chính chúng.

## Cổng (đăng ký trước)

Đo trên toàn bộ audit2, bootstrap theo tài liệu với 2.000 lần lặp. So sánh ghép cặp trên cùng (tài liệu, field) dùng `paired_systems`.

| | Điều kiện để chuyển serving từ S sang X ∈ {F, M} |
|---|---|
| N1, bắt lỗi | recall lỗi của X ≥ **95%** (điểm ước lượng) **và** ≥ recall của S − **1,0 điểm** |
| N2, review | cận trên khoảng 95% của (review X − review S) ≤ **+1,5 điểm** |

Quyết định: serve **F** nếu F qua cả N1 và N2. Nếu không, serve **M** nếu M qua cả hai. Nếu không nữa thì giữ **S**.

Báo cáo kèm, không phải cổng:
- Recall và review của S trên audit2. Nếu recall của S dưới 95% thì ngưỡng bảo thủ không chuyển được sang các form này; ghi nhận như một cảnh báo calibration, còn cổng N1/N2 vẫn tính như trên.
- **Luật dạng, số sạch:** S so với R trên cùng trace: review, recall, số field bị gắn cờ và số cờ đúng là lỗi. Kỳ vọng ghi trước: review giảm khoảng 2 điểm, recall không giảm quá 1 điểm. Nếu không đạt, cân nhắc gỡ luật.
- Tốc độ (giây/tài liệu) lúc cache audit2. GPU dùng chung nên chỉ để tham khảo. Số chính là `scripts/compare_decode.py`.
- Độ khớp value giữa các hệ thống trên audit2.

## Lệnh

```bash
C=configs/confidence/kev.yaml; P=artifacts/kev-r4/confidence/audit2/cohort_plan.json
# S: cache mới chỉ chứa audit2; các cohort phát triển trỏ tới cache gốc
l2r4kie cache-traces --config $C --split audit --set cohort_plan=$P --set cache=artifacts/kev-r4/confidence/cache-audit2
l2r4kie audit --config $C --set cohort_plan=$P --set cache=artifacts/kev-r4/confidence/cache-audit2 \
    --set output=artifacts/kev-r4/confidence/selected-shapes
# M, F: cùng cách, với merge_adapter=true (và fast_decode=true), so ghép cặp với S
l2r4kie audit --config $C ... --set output=artifacts/kev-r4/confidence/selected-merged \
    --baseline-predictions 'artifacts/kev-r4/confidence/selected-shapes/heads/attention[value@14+value@21]/audit_predictions.jsonl'
```

## Kết quả

Chưa chạy.
