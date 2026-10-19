# Thử nghiệm: độ đồng thuận giữa các ứng viên có giúp head confidence không?

Trạng thái: **đã chạy giai đoạn 1: âm tính, dừng** (review −0,31 điểm, cần ≥ 1,5). Giao thức viết ngày 2026-10-18, trước khi sample dev. Phần "Giao thức" và "Luật quyết định" không được sửa sau khi có số. Kết quả điền vào cuối.

## Câu hỏi

Trên `kev_r4`, head chính (`attention[value@14+value@21]`) review 30,4% ở mức bắt 96,9% lỗi trên audit, ngang r4 + attention. Mức sàn oracle là 13,1%. Head chỉ đọc trace của một lượt decode greedy.

Câu hỏi: nếu cho model viết lại mỗi field thêm K lần (sampling), thì tỷ lệ ứng viên trùng với value greedy có thêm thông tin mà head chưa có không? Đây là tín hiệu mới, không phụ thuộc vị trí marker, và gần với tinh thần Kev (so nhiều ứng viên).

## Giao thức

**Sinh ứng viên** (`scripts/agreement_signal.py sample`, `l2r4kie.model.decode.sample_prefix`):

| | Giá trị |
|---|---|
| Ứng viên mỗi field | K = 8 |
| Nhiệt độ | 1,0, trên cùng logits đã chặn special token như greedy |
| Budget mỗi ứng viên | 512 token, early stop bật |
| Seed | 0; seed mỗi tài liệu = sha256(`0:<document_id>`), không phụ thuộc thứ tự |
| Field | đúng các field trong trace cache của cohort (24 field đầu mỗi tài liệu) |

K, nhiệt độ và budget được chọn theo chi phí (smoke 3 tài liệu: khoảng 5 s/tài liệu), trước khi có bất kỳ số đo chất lượng nào.

**Feature của một field** (chỉ field greedy `ok`; field khác vẫn bắt buộc review như cũ):
- `agreement`: tỷ lệ ứng viên `ok` và bằng value greedy theo comparator `text`.
- `distinct`: số value khác nhau trong K ứng viên (mỗi status không `ok` tính là một value), chia cho K.
- `logit`: logit đã calibration của head chính đã đóng băng (`frozen.json`).

**Bốn hệ thống, cùng các field:**

| Tên | Điểm |
|---|---|
| `head` | head chính, calibration đã đóng băng |
| `head_cv` | hồi quy logistic chỉ trên `logit`, CV |
| `agreement` | `agreement` một mình (hòa điểm giữ nguyên) |
| `stack` | hồi quy logistic trên `logit`, `agreement`, `distinct`, CV |

- Hồi quy logistic: feature chuẩn hóa, L2 = 0,001, L-BFGS.
- CV 5 fold **theo tài liệu** (seed 0). Mỗi field được chấm bởi model không thấy tài liệu của nó. `head_cv` là đối chứng công bằng cho `stack`: cùng quy trình CV, chỉ thiếu feature đồng thuận.
- Review ở mức bắt 95% lỗi: `fit_policy` trên điểm gộp của cả cohort, như khi chọn head.
- So sánh ghép cặp: `bootstrap_comparison` (bootstrap theo tài liệu, 2.000 lần, seed 0).

**Giai đoạn 1 (sàng lọc): cohort dev** (247 tài liệu, 4.489 field).
- Dev đã được dùng để chọn head chính, nên số của `head` trên dev lạc quan. `stack` dùng chính logit đó, nên cũng thừa hưởng phần lạc quan này. So sánh chính là `stack` với `head_cv`, hai bên cùng chịu.
- Audit (347 tài liệu) **không được dùng** trong thử nghiệm này.

## Luật quyết định (giai đoạn 1)

- **Tiếp tục sang giai đoạn 2** khi cả hai điều kiện đúng:
  1. Review của `stack` thấp hơn `head_cv` ít nhất **1,5 điểm** (điểm ước lượng).
  2. Khoảng 95% bootstrap của chênh lệch review (`stack` − `head_cv`) nằm hoàn toàn dưới 0.
- **Ngược lại:** dừng, ghi kết quả âm tính. Không thử K, nhiệt độ hay feature khác trên cùng dev để "cứu" kết quả.
- Báo cáo kèm, không phải luật: `agreement` một mình, AUROC, AURC, AUROC trong cùng (form, field), chi phí sampling mỗi tài liệu.

## Giai đoạn 2 (chỉ khi giai đoạn 1 đạt; viết chi tiết trước khi chạy)

- Sample calibration và risk_validation với cùng cài đặt.
- Fit `stack` trên toàn bộ dev, calibration trên cohort calibration, ngưỡng bảo thủ trên risk_validation (như `finalize`).
- Đo một lần trên một **cohort audit mới**: tài liệu không thuộc tập train của extractor và không thuộc 5 cohort hiện có. Cổng (không-kém-hơn và hơn hẳn so với head chính trên cùng audit mới) sẽ được ghi vào đây trước khi decode audit mới.

## Kết quả giai đoạn 1 (dev, chạy 2026-10-18)

File: `artifacts/kev-r4/agreement/dev.{jsonl,settings.json,analysis.json}`.
- 247 tài liệu, 4.489 field, 571 lỗi. Mức sàn oracle 12,1%.
- Lần chạy đầu với 96 nhánh mỗi chunk hết bộ nhớ GPU ở tài liệu thứ 10 (GPU dùng chung). Output dở đã bị xóa và chạy lại từ đầu với `--max-branches 32`. Đây là cài đặt kỹ thuật, không đổi giao thức.

| Hệ thống | Review @95% | Bắt lỗi | AUROC | AURC | AUROC trong field |
|---|---:|---:|---:|---:|---:|
| `head` | 24,28% | 95,1% | 0,962 | 0,0133 | 0,820 |
| `head_cv` | 24,24% | 95,1% | 0,962 | 0,0134 | 0,817 |
| `agreement` | 100% | 100% | 0,892 | 0,0293 | 0,750 |
| **`stack`** | **23,93%** | 95,1% | 0,963 | 0,0132 | 0,823 |

**Luật quyết định:** `stack` − `head_cv` = **−0,31 điểm** [KTC 95%: −0,63; −0,02].
- Khoảng tin cậy nằm dưới 0: điều kiện 2 đạt.
- Mức giảm nhỏ hơn nhiều so với 1,5 điểm: điều kiện 1 **trượt**.
- **Kết luận: dừng, không sang giai đoạn 2.** Độ đồng thuận có thêm một chút thông tin thật, nhưng quá nhỏ để đáng giá.

**Vì sao (mô tả, sau khi đã có kết quả; không dùng để quyết định):**
- 56,9% field có cả 8 ứng viên trùng value greedy, và nhóm này đúng 98,8%. Nhóm còn lại đúng 74,3%. Vậy độ đồng thuận có liên quan rõ tới đúng/sai.
- Nhưng **head đã biết gần hết điều đó.** Trong mô hình `stack`, trọng số chuẩn hóa của logit head khoảng 2,4, của `agreement` khoảng 0,45, của `distinct` khoảng 0,25 (ổn định qua 5 fold). Sampling chủ yếu đo lại độ bất định ở từng token, mà head đã đọc được từ trace.
- `agreement` một mình phải review 100%: 30 lỗi (5,3% số lỗi) có cả 8 ứng viên trùng nhau, nhiều hơn 5% được phép bỏ sót. Đây là lỗi tự tin: model đọc nhầm ổn định, hoặc ground truth khác cách viết (`35°` so với `35º`, `10/01/2023` so với `10 /01 /2023`). Head chính bắt được 23/30 lỗi loại này ở ngưỡng dev, và 471/492 lỗi còn lại.
- Chi phí: sampling mất trung bình 16,7 s/tài liệu (trung vị 14,5 s, tối đa 59 s), cỡ bằng một lượt decode greedy nữa. Đổi lại chỉ giảm 0,3 điểm review.

**Ghi chú cho lần sau.** Tài liệu chưa dùng cho audit mới: còn khoảng 1.090 (train_reserve 401, dev 339, calibration 251, test 101), sau khi bỏ 5 cohort, 1.997 tài liệu bị loại trong plan của r4, tài liệu eval của extractor và tài liệu train. Nhưng chúng chỉ thuộc 16/58 form, nên một audit mới từ nguồn này sẽ lệch form so với audit cũ.
