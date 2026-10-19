# Báo cáo phương pháp L2R4KIE: trích xuất KevFormat và review theo độ tin cậy

Ngày: 2026-10-19.

## Tóm tắt

L2R4KIE trích xuất field từ ảnh tài liệu bằng Qwen2-VL-2B + LoRA, rồi dùng một head confidence đọc trace của chính lượt decode để quyết định field nào cần người review. Trên audit 347 tài liệu, hệ thống bắt **96,9%** lỗi với **30,4%** field phải review. Mức này ngang hệ cũ r4 (29,9%), khác biệt nằm trong nhiễu.

Những điểm chính:

- **Format KevFormat** (mỗi field là một nhánh decode riêng trên prefix ảnh dùng chung) giữ độ chính xác của r4: EM scalar dev 89,8% so với 90,3%, khoảng tin cậy chứa 0. Cắt cụt giảm khoảng 8 lần (2,29% xuống 0,28%).
- **So với xuất JSON thông thường** (cùng model, cùng công thức train): nhanh hơn **2,5×** (1.976 s so với 4.858 s cho 116 tài liệu) và EM cao hơn **8,5 điểm** trên dev.
- **Luật dạng value** đưa vào review bắt buộc những value có định dạng field chưa từng có. Review giảm khoảng 2 điểm (29,2% xuống 27,2% trên risk_validation), mức bắt lỗi giữ nguyên. Đây đang là bundle được serve (`selected-shapes`).
- **Tối ưu tốc độ infer** không cần train lại: merge LoRA nhanh **2,1×**; CUDA graph cho đuôi decode nhanh thêm **1,7× (24 tài liệu dev, 98,7% value trùng**, GPU dùng chung).
- **Đã thử và loại:** tín hiệu đồng thuận từ sampling chỉ giảm review 0,3 điểm, trong khi tốn thêm một lượt decode.

**Đang chạy:** audit mới 318 tài liệu, có cổng đăng ký trước, để có số sạch cho luật dạng và quyết định có chuyển serving sang bản merge hoặc CUDA graph hay không.

## Bài toán và tiêu chí

Mục tiêu là bắt ít nhất 95% field sai mà phải review càng ít field càng tốt. Mốc so sánh là hệ r4 (JSON, budget 256 token, head attention): review 29,9%, bắt 96,8% lỗi.

- **Đầu vào:** ảnh tài liệu (form hành chính, hóa đơn, bảng kê, ...) và danh sách field cần lấy, mỗi field có id, mô tả và loại (scalar hoặc mảng).
- **Đầu ra:** value của từng field, kèm quyết định *chấp nhận tự động* hoặc *gửi người review*, và lý do.
- **Dữ liệu:** 58 form. Mỗi tài liệu lấy 24 field đầu khi đo, như r4.

| Cổng | Điều kiện | Đo trên |
| --- | --- | --- |
| G1 | EM scalar ≥ r4 − 1 điểm | dev extractor (116 tài liệu) |
| G2 | output tọa độ < 0,5% | dev extractor |
| G3 | cắt cụt ≤ r4 (2,29%) | dev extractor |
| G4a | review ≤ 29,9% | audit 347 tài liệu, sau khi đóng băng |
| G4b | bắt lỗi ≥ 95% | audit 347 tài liệu |

Mọi cổng đều được viết trước khi có số, và không sửa sau khi đã thấy kết quả. Audit chỉ mở một lần, sau khi head, calibration và ngưỡng đã đóng băng; code từ chối chạy audit lần hai.

## Extractor KevFormat

Ảnh được encode một lần thành prefix, rồi mỗi field được decode thành một nhánh độc lập trên KV cache của prefix đó. Các nhánh không thấy nhau, nên lỗi ở một field không lan sang field khác và mọi field chạy song song.

**Cấu trúc một nhánh** (sau prefix ảnh + hướng dẫn):

```
<|object_ref_start|> mô tả field <|object_ref_end|> <|box_start|> value dạng text <|box_end|>
                                    ↑ key            ↑ decide                   ↑ value
```

- Value viết dạng text, không phải JSON. Mảng viết dạng JSON array bên trong marker.
- `<|box_end|>` đóng value. Ba vị trí key, decide, value là nơi head confidence đọc state (phần sau).
- Vị trí M-RoPE của mọi nhánh đều nối tiếp ngay sau prefix, giống hệt lúc train. Nhờ vậy có thể thay engine decode (vLLM) mà không train lại.

**Train** (giống công thức r4 để so công bằng):

| Thành phần | Giá trị |
| --- | --- |
| Model | Qwen2-VL-2B-Instruct, bf16 |
| LoRA | r16, α 32, trên q/k/v/o |
| Ảnh | tối đa 2,1 MP |
| Field mỗi tài liệu khi train | 12, lấy mẫu |
| Step, learning rate, seed | 1.000, 1e-4, 42 |
| Loss | chỉ cross-entropy trên token value (kể cả `box_end`) |

**Decode khi serve** (`model/decode.py`):

1. Encode ảnh, prefill prefix một lần.
2. Chép KV của prefix cho tối đa 64 nhánh mỗi đợt, prefill prompt của các nhánh.
3. Greedy từng bước cho cả batch. Special token bị chặn, trừ marker đóng.
4. Nhánh dừng khi gặp `box_end`, khi hết budget 16.384 token, hoặc khi **early stop** thấy nó không thể còn `ok` (ví dụ mảng đã sai cú pháp JSON).
5. Khi một nửa số nhánh đã xong, batch được thu gọn để các nhánh còn lại chạy nhanh hơn.
6. Parse value theo loại field. Status: `ok`, `truncated`, `invalid_array`, ...

## Tín hiệu và head confidence

Head chính là `attention[value@14+value@21]`. Nó đọc state của chính lượt decode đã sinh ra value, không cần thêm lượt encode ảnh hay model verifier nào.

**Tín hiệu lưu trong trace của mỗi field:**

| Tín hiệu | Lấy ở đâu | Ý nghĩa |
| --- | --- | --- |
| `value` | state sau khi đưa `box_end` vào | model đã đọc xong toàn bộ value |
| `key` | `object_ref_end` | đã đọc tên field, chưa viết value |
| `decide` | `box_start` | ngay trước khi viết value |
| `value@14`, `value@21` | state `value` ở layer 14 và 21 | layer giữa còn giữ thông tin về nội dung |
| token state | state sau từng token value (tối đa 512) | cho attention pooling |
| thống kê token | log-prob, entropy, margin, prob, log-prob của `box_end` | 5 cột mỗi token |

**Head chính:** attention pooling trên token state, ghép với state `value` ở layer 14 và 21, ra một logit "value này đúng". Train bằng cross-entropy (có biến thể thêm loss xếp hạng), trên nhãn đúng/sai so với ground truth.

**Chọn head** (luật đăng ký trước):

- Grid 68 tổ hợp × 3 seed × 3 snapshot (75, 150, 300 step). Các family: `end`, `mean`, `hybrid`, `hybrid_mlp`, `attention`, `query` (query là `key`), thêm prior theo field, và hai heuristic không train (min / mean log-prob).
- Mỗi family giữ snapshot tốt nhất trên dev. Head chính là family có review thấp nhất ở mức bắt 95% lỗi trên dev; hòa thì xét AURC, rồi AUROC.

**Cohort** (không tài liệu nào thuộc tập train của extractor; code tự kiểm tra):

| Cohort | Tài liệu | Field | Dùng để |
| --- | --: | --: | --- |
| train | 796 | 14.895 | train head |
| dev | 247 | 4.489 | chọn head |
| calibration | 158 | 2.986 | fit calibration |
| risk_validation | 332 | 6.224 | chọn ngưỡng bảo thủ |
| audit | 347 | 6.580 | đo một lần, sau khi đóng băng |

Qua ablation: state ở layer giữa tốt hơn layer cuối (review thấp hơn 1,2 điểm với cùng head). Thêm `key` và `decide` không giúp mà còn làm review tăng.

## Calibration, ngưỡng và policy review

Một field được chấp nhận tự động chỉ khi nó qua hết các lý do review bắt buộc và có confidence ≥ ngưỡng. Ngưỡng của bundle đang serve là **0,9220**.

**Thứ tự xét lý do review** (lý do đầu tiên khớp được ghi lại):

1. `field_outside_evaluated_schema`: field chưa có trong dữ liệu đã đánh giá.
2. `incomplete_or_invalid`: value bị cắt cụt, mảng sai cú pháp, ...
3. `field_shape_violation`: value có dạng mà field chưa từng có (luật dạng, bên dưới).
4. `low_correctness_confidence`: confidence thấp hơn ngưỡng.

**Calibration:** fit trên cohort calibration (158 tài liệu), chuyển logit của head thành xác suất đúng. Trên audit, ECE là 0,012.

**Ngưỡng bảo thủ:** chọn trên risk_validation (332 tài liệu), là ngưỡng thấp nhất mà **cả hai** cận dưới của recall lỗi đều ≥ 95%:

- cận dưới Wilson 95%;
- cận dưới bootstrap 95% theo tài liệu (2.000 lần lặp).

Nhờ vậy, mục tiêu 95% được giữ cả khi lỗi trong một tài liệu tương quan với nhau.

**Luật dạng value** (`confidence/shapes.py`):

- Head bỏ sót một nhóm lỗi rất tự tin: thiếu hoặc thừa chữ số, đọc nhầm dấu phân cách (`10/01/2023` thành `10 /01 /2023`, `35°` thành `35º`).
- **Dạng** của một value: chữ số thành `9`, chữ cái thành `a`, ký tự khác giữ nguyên. Ví dụ `12/03/2024` có dạng `99/99/9999`.
- **Học luật** chỉ từ nhãn tập train của extractor. Mỗi field scalar có chữ số và có ≥ 20 value sẽ giữ các dạng chiếm ≥ 2%, tối đa 5 dạng, và các dạng giữ lại phải phủ ≥ 95% số value. Kết quả có 528 field có luật.
- **Áp luật:** value `ok` có dạng ngoài tập cho phép thì bắt buộc review, dù confidence cao đến đâu. Ngưỡng sau đó được fit lại trên các field còn lại: 0,9362 xuống 0,9220.
- Luật không tốn GPU: chỉ là so khớp chuỗi trên value đã decode.

**An toàn khi serve:** bundle ghi fingerprint của mọi file (head, calibration, policy, luật dạng) và cách decode (`merge_adapter`, `fast_decode`). `infer` và `audit` từ chối khi có file bị sửa hoặc trace được decode khác với lúc fit ngưỡng. Test parity kiểm tra rằng confidence và quyết định lúc serve trùng với lúc audit.

## Kết quả

Extractor qua G1–G3 và ngang r4 về độ chính xác. Confidence bắt 96,9% lỗi nhưng review 30,4%, vượt mốc G4a 0,53 điểm; khoảng tin cậy so với r4 chứa 0.

### Độ chính xác extractor (116 tài liệu mỗi split, 24 field đầu)

| Metric | r4 dev | Kev dev | r4 test | Kev test |
| --- | --: | --: | --: | --: |
| EM scalar | 90,33% | **89,79%** | 92,20% | **91,37%** |
| EM mảng | 19,0% | **22,0%** | 21,4% | **22,4%** |
| EM tất cả field | 87,00% | **86,62%** | 88,98% | **88,23%** |
| Cắt cụt | 2,29% | **0,28%** | 2,19% | **0,28%** |
| Output tọa độ | 0% | **0%** | 0% | **0%** |

- Kev − r4 trên EM scalar: −0,54 điểm dev [−1,90; +0,79], −0,83 test [−2,09; +0,35]. Cả hai khoảng đều chứa 0.
- Cắt cụt giảm khoảng 8 lần nhờ budget 16.384 token thay vì 256.
- Điểm yếu thấy rõ: lỗi dấu và hoa/thường nhiều hơn r4 (59 so với 49 trên dev).

### Cổng G4 trên audit (347 tài liệu, 6.580 field, 905 lỗi)

|  | Kev | Mốc | r4 cùng audit | Kết luận |
| --- | --: | --: | --: | --- |
| G4a review | 30,43% [28,6; 32,2] | ≤ 29,9% | 29,86% | trượt 0,53 điểm |
| G4b bắt lỗi | 96,9% [95,8; 97,9] | ≥ 95% | 96,8% | đạt |

- So ghép cặp với r4 (bootstrap theo tài liệu): Kev − r4 = +0,56 điểm review [−0,72; +1,82]. Hai hệ ngang nhau.
- AUROC 0,966, ECE 0,012. Mức sàn lý thuyết (review đúng các field sai) là 13,1%; head đang review gấp khoảng 2,3 lần mức đó.
- Theo ngân sách review: 10% bắt 63,8% lỗi; 20% bắt 89,7%; 30% bắt 96,7%; 50% bắt 99,1%.
- Bundle này (`selected`) chưa có luật dạng; audit này đo trước khi có luật.

### Luật dạng value (head chính, mục tiêu bắt 95%)

| Cohort | Cờ (sai / tổng) | Review: không luật → có luật | Bắt lỗi |
| --- | --: | --- | --- |
| risk_validation (802 lỗi) | 98 / 99 | 29,18% → **27,23%** | 96,51% → 96,51% |
| dev (571 lỗi) | 62 / 62 | 28,25% → **26,13%** | 96,85% → 97,02% |

- 160/161 field bị gắn cờ là lỗi thật.
- Mục tiêu bắt lỗi càng cao thì luật càng có lợi: ở mức 99%, review trên risk_validation giảm từ 42,8% xuống 36,5%.
- Số này chưa sạch: tham số của luật được đặt khi xem lỗi trên chính risk_validation và dev. Audit mới sẽ cho số sạch.

## So với infer JSON thông thường

Cùng model, LoRA, tài liệu, field lấy mẫu, số step và loss, KevFormat **nhanh hơn 2,5×** và **chính xác hơn khoảng 8 điểm EM** so với cách để model viết mọi field thành một object JSON trong một chuỗi.

**Cách đo** (`scripts/json_baseline.py`, commit `b913ade`):

- Baseline JSON: model viết các field được hỏi thành một object JSON, mỗi request 12 field (bằng cỡ nhóm lúc train), 2 request mỗi tài liệu.
- Hai bên cùng 116 tài liệu dev, 24 field đầu, budget 16.384 token, early stop, LoRA chưa merge, chạy eager, cùng GPU, không trace.

|  | JSON một chuỗi | KevFormat | Lợi |
| --- | --: | --: | --: |
| Tổng thời gian 116 tài liệu | 4.858 s | **1.976 s** | 2,5× |
| Trung vị giây / tài liệu | 14,1 | **4,1** | 3,4× |
| p90 giây / tài liệu | 49,0 | **43,9** | 1,1× |
| Tài liệu chậm nhất | 1.034 s | **372 s** |  |
| EM dev | 78,1% | **86,6%** | +8,5 điểm [5,9; 11,4] |
| EM test | 80,7% | **88,2%** | +7,5 điểm [5,6; 9,5] |
| Field bị sót (`missing`, dev) | 155 | **0** |  |

- Kev nhanh hơn ở 97/116 tài liệu.
- Bỏ riêng tài liệu tệ nhất của JSON (một object chạy hết budget, 16.691 token, 1.034 s; Kev xử lý tài liệu này trong 6 s), Kev vẫn nhanh hơn 1,9×.

**Vì sao nhanh hơn:** JSON phải sinh tuần tự mọi field cộng cú pháp JSON trong một chuỗi (trung vị khoảng 600 token mỗi tài liệu). Kev sinh 24 nhánh song song, và mỗi bước decode tốn gần như nhau dù có 1 hay 64 nhánh (73–77 ms/bước khi chưa merge), nên thời gian chỉ bằng nhánh dài nhất.

**Lưu ý để đọc đúng:**

- Khoảng cách 8 điểm là so với baseline JSON *cùng công thức train với Kev*. Hệ r4 cũng xuất JSON nhưng cấu hình khác, và đạt EM 87,0%, ngang Kev. Vì vậy kết luận chắc nhất phần chính xác là: Kev không kém JSON, và hơn hẳn khi gom nhiều field vào một chuỗi.
- Merge LoRA và CUDA graph (phần sau) cũng áp dụng được cho JSON. Con số 2,5× là lợi ích riêng của format; các tối ưu sau đó là độc lập.
- Baseline JSON không có head confidence tương đương, nên chưa so được ở chế độ có confidence.

## Tốc độ infer và tối ưu

Thời gian nằm gần hết ở đuôi decode, khi chỉ còn 1–2 field dài (thường là bảng). Merge LoRA (2,1×) và CUDA graph cho đuôi (thêm 1,7×) cùng nhắm vào đó, và không cần train lại.

### Thời gian đi đâu

Đo trên 24 tài liệu dev, toàn bộ field (667 field), trace bật, LoRA chưa merge (`scripts/profile_infer.py`). Trung vị 14,0 s/tài liệu, p90 59,5 s.

| Phần | Tỷ lệ thời gian |
| --- | --: |
| Đọc ảnh và processor | 2,7% |
| Prefill prefix và prompt của nhánh | 1,5% |
| **Các bước decode** | **92,2%** |
|  trong đó khi chỉ còn ≤ 2 nhánh chạy | 80,1% |
| Phần việc trên host (lm_head, argmax, vòng Python) | 3,5% |

Mỗi bước decode bị chặn bởi overhead launch kernel, không phải bởi tính toán. 1 nhánh hay 64 nhánh tốn gần như nhau:

| Cấu hình (KV 2.678 token) | 1 nhánh | 16 nhánh | 64 nhánh |
| --- | --: | --: | --: |
| LoRA chưa merge, eager | 73 ms | 74 ms | 77 ms |
| Đã merge LoRA, eager | 11–21 ms | 11 ms | 18 ms |
| Đã merge + CUDA graph | **3,2 ms** | 6,9 ms | 19 ms |

### Merge LoRA (`merge_adapter`)

Gộp LoRA vào trọng số gốc sau khi load, nên mỗi linear bớt 2 matmul nhỏ. Đo trên 24 tài liệu, 434 field, so với cache gốc:

|  | Chưa merge | Đã merge |
| --- | --: | --: |
| Giây / tài liệu | 17,9 | **8,6** (2,1×) |
| Value trùng cache gốc | 98,4% | 97,5% |
| Lệch confidence p99 / max | 0,017 / 0,050 | 0,091 / 0,337 |
| Quyết định review bị đổi | 1 | 0 |

Numerics bf16 đổi nhẹ, nên phải cache lại trace bằng bản đã merge và fit lại ngưỡng trước khi serve.

### CUDA graph cho đuôi (`fast_decode`)

`model/fast_decode.py` tách decode thành hai pha:

1. **Sinh token.** Cùng greedy, cùng chặn special token, cùng early stop như decode thường. Khi batch còn rộng thì chạy eager. Khi còn ≤ 4 nhánh thì chuyển sang StaticCache với bước decode đã `torch.compile(mode='reduce-overhead')`. Mỗi cỡ (số dòng 1/2/4 × độ dài cache 4.096 đến 32.768) có một graph riêng.
2. **Lấy tín hiệu.** Một lượt forward teacher-forced cho prompt cộng token đã sinh của mọi nhánh, rồi đọc state key/decide/value, layer 14/21 và thống kê token. Trong FP32, kết quả trùng hoàn toàn với decode thường (6 test).

Warm up 12 graph mất khoảng 270 s, làm một lần khi server khởi động. So sánh trên 24 tài liệu dev, toàn bộ field (667 field), A và B chạy xen kẽ trên cùng tài liệu:

|  | Merge, eager | Merge + CUDA graph |
| --- | --: | --: |
| Tổng | 861 s | **500 s** (1,72×) |
| Trung vị giây / tài liệu | 22,3 | **12,2** |
| Status trùng |  | 100% |
| Value trùng |  | 98,7% |
| Lệch confidence trung vị / p99 / max |  | 0,0007 / 0,049 / 0,101 |
| Quyết định review bị đổi |  | 2 / 667 |

- Lợi lớn nhất ở tài liệu có đuôi dài, ví dụ `tsr-a02`: 70,6 s xuống 20,0 s (3,5×).
- Ở tài liệu nhiều field ngắn, fast có thể chậm hơn, vì lượt teacher-forced tốn hơn phần CUDA graph tiết kiệm được.
- Số tuyệt đối bị phóng to vì GPU đang chạy 8 tiến trình cùng lúc. Tỷ lệ giữa A và B đáng tin hơn.

### Các hướng còn lại

| Hướng | Lợi dự kiến | Chi phí |
| --- | --- | --- |
| Batch nhiều tài liệu cùng lúc | throughput 5–15× khi có hàng đợi | scheduler riêng, hoặc vLLM |
| vLLM sinh value, HF teacher-forced lấy tín hiệu | CUDA graph, paged KV, prefix caching, continuous batching | lớn; pha teacher-forced đã có sẵn trong `fast_decode` |
| Chồng đọc ảnh lên decode tài liệu trước | ≤ 2,7% | thấp |
| Chia bảng thành một nhánh mỗi dòng | rút chính đuôi dài | train lại, đo lại toàn bộ |

## Thí nghiệm âm tính và giới hạn của head

Cho model viết lại mỗi field nhiều lần để đo độ đồng thuận chỉ giảm review 0,31 điểm, trong khi cần ≥ 1,5 điểm mới đáng làm. Hướng này đã dừng.

### Tín hiệu đồng thuận (`agreement_signal.md`)

- **Cách làm:** 8 ứng viên mỗi field, nhiệt độ 1,0, budget 512 token. Feature: tỷ lệ ứng viên trùng value greedy, số value khác nhau. Ghép với logit của head bằng hồi quy logistic, CV 5 fold theo tài liệu.
- **Luật quyết định** (viết trước khi có số): đi tiếp nếu review giảm ≥ 1,5 điểm và khoảng tin cậy nằm dưới 0.

| Hệ thống (dev, 247 tài liệu, 571 lỗi) | Review @95% | AUROC |
| --- | --: | --: |
| head (logit, CV) | 24,24% | 0,962 |
| chỉ độ đồng thuận | 100% | 0,892 |
| head + đồng thuận | **23,93%** | 0,963 |

- Chênh lệch: −0,31 điểm [−0,63; −0,02]. Có thông tin thật nhưng quá nhỏ.
- Lý do: head đã đọc được gần hết độ bất định ở từng token. Trong mô hình ghép, trọng số chuẩn hóa của logit head khoảng 2,4, của độ đồng thuận khoảng 0,45.
- Chi phí: trung bình 16,7 s/tài liệu, cỡ một lượt decode nữa.

### Muốn bắt 100% lỗi có cần train lại head không?

Không. Giới hạn nằm ở **lỗi tự tin**, không ở head:

- 30 lỗi trên dev (5,3% số lỗi) có cả 8 ứng viên trùng nhau. Model đọc nhầm một cách ổn định, hoặc nhãn viết khác cách (`35°` so với `35º`). Tín hiệu bất định không phân biệt được những lỗi này.
- Chỉ có thể hạ ngưỡng thì mới bắt thêm. Giá tăng rất nhanh: trên risk_validation, bắt 95% cần review 24,7%, bắt 99% cần 36,5% (đã có luật dạng).
- Với mẫu hữu hạn, không thể chứng minh thống kê mức 100% nếu không review tất cả.
- Cách hiệu quả hơn là thêm tín hiệu độc lập với model, đúng kiểu luật dạng value: nó bắt đúng nhóm lỗi tự tin mà head bỏ sót.

## Audit mới (audit2)

Audit2 gồm 318 tài liệu chưa từng được dùng ở đâu, với cổng được đăng ký trước khi decode. Nó trả lời hai câu: luật dạng có giữ được mức giảm khoảng 2 điểm trên dữ liệu mới không, và có nên chuyển serving sang bản nhanh hơn không. **Chưa có kết quả.**

**Cohort** (`scripts/plan_fresh_audit.py`, seed 211):

- 318 tài liệu, 16 form, tối đa 20 tài liệu mỗi form, khoảng 6.286 field.
- Đã loại: mọi cohort trước đây (gồm audit cũ), 1.997 tài liệu trong danh sách loại của r4 (đã gồm tài liệu eval của extractor), và tập train của extractor.
- Giới hạn: chỉ 16/58 form còn tài liệu chưa dùng, nên audit2 không thay được G4.

**Ba hệ thống** (cùng head; mỗi hệ fit lại calibration, ngưỡng và luật dạng trên trace của chính nó):

| Tên | Bundle | Decode |
| --- | --- | --- |
| S (đang serve) | `selected-shapes` | LoRA chưa merge, eager |
| M | `selected-merged` | merge LoRA |
| F | `selected-fast` | merge LoRA + CUDA graph |

**Cổng để chuyển serving từ S sang X** (ghép cặp trên cùng tài liệu và field, bootstrap 2.000 lần theo tài liệu):

|  | Điều kiện |
| --- | --- |
| N1, bắt lỗi | recall của X ≥ 95% **và** ≥ recall của S − 1,0 điểm |
| N2, review | cận trên khoảng 95% của (review X − review S) ≤ +1,5 điểm |

Quyết định: serve F nếu F qua cả hai. Nếu không thì serve M nếu M qua cả hai. Nếu không nữa thì giữ S. Báo cáo kèm: số sạch của luật dạng (S so với cùng head không luật, trên cùng trace), và độ khớp value giữa các hệ.

**Tiến độ** lúc 08:41 UTC (GPU dùng chung, 8 tiến trình):

| Job | Tiến độ |
| --- | --- |
| Cache audit2 cho S | 3/20 phần |
| Cache lại bản M (dev, calibration, risk_validation, train) | dev 15/16, calibration 6/10, còn risk_validation và train |
| Cache lại bản F | dev 9/16 |

**Đã tạm dừng** lúc 08:50 UTC, trước khi có kết quả. Các phần cache đã xong vẫn được giữ. Khi chạy lại cùng lệnh, worker sẽ tự nhận lại các lock cũ và chạy tiếp từ phần dở.

## Giới hạn, rủi ro và bước tiếp theo

### Giới hạn đã biết

- **G4a trượt 0,53 điểm.** Trên audit cũ, mức giảm review so với r4 thấp hơn mục tiêu 0,53 điểm. Recall vẫn đạt, nhưng chưa chứng minh được đầy đủ là review giảm đủ nhiều.
- **Số của luật dạng chưa sạch.** Luật được thiết kế sau khi xem lỗi trên dev và risk_validation, nên mức giảm khoảng 2 điểm review có thể lạc quan. Audit2 là lần đo sạch đầu tiên.
- **Audit2 chỉ phủ 16/58 form.** Chỉ các form này còn tài liệu chưa dùng, nên kết quả audit2 không thay được G4 cho toàn bộ form.
- **Review theo tài liệu vẫn gần 100%.** Hầu như tài liệu nào cũng có ít nhất một field cần review. Lợi ích nằm ở số field phải xem, chưa ở số tài liệu được bỏ qua hoàn toàn.
- **Số tốc độ có nhiễu.** GPU dùng chung với job khác. Số chính lấy từ `compare_decode.py` (chạy hai bên trên cùng tài liệu và cùng lúc), còn thời gian lúc cache chỉ để tham khảo.
- **So sánh với JSON có hai lưu ý.** (1) r4, cũng xuất JSON nhưng setup khác, đạt 87,0% EM. (2) Merge LoRA và CUDA graph cũng áp dụng được cho JSON, nên tỷ lệ 2,5× là so trên cùng cách decode eager, không phải so bản tối ưu nhất của mỗi bên.
- **Lỗi dấu tiếng Việt** vẫn là nhóm lỗi value lớn. Luật dạng không bắt được chúng vì value sai dấu vẫn đúng dạng.

### Rủi ro khi serve

- **Lệch numerics giữa decode và trace.** Ngưỡng chỉ đúng với đúng cách decode đã dùng khi fit. Các cờ `merge_adapter` và `fast_decode` được ghi vào cache, `bundle.json` và `frozen.json`. `finalize`, `infer` và `audit` từ chối chạy nếu chúng không khớp.
- **Thời gian warm-up của fast_decode.** Lần đầu cần khoảng 265–300 giây để dựng 12 graph. Chỉ đáng dùng khi tiến trình serve sống lâu.
- **Bộ nhớ GPU.** StaticCache cấp sẵn theo bucket độ dài, tối đa 32.768 token, nên tốn bộ nhớ hơn eager.
- **Drift dữ liệu.** Form mới hoặc ảnh chất lượng khác có thể làm calibration lệch. Cần audit định kỳ trên tài liệu mới.

### Bước tiếp theo

1. Chờ audit2 xong, điền kết quả vào `docs/results/fresh_audit.md`, áp cổng N1/N2 và chọn bundle serve (F, M hoặc giữ S).
2. Nếu luật dạng không giữ được mức giảm review trên audit2, gỡ luật và quay lại ngưỡng 0,9362.
3. Thu thêm tài liệu mới cho 42 form còn lại để có audit phủ đủ form.
4. Xử lý lỗi dấu: thêm tín hiệu ở mức ký tự, hoặc kiểm tra theo từ điển cho các field tên riêng và địa chỉ.
5. Nếu cần nhanh hơn nữa, thử batch nhiều tài liệu chung một lượt prefill, hoặc chuyển sang vLLM cho phần decode (vẫn cần pass teacher-forced để lấy trace).

## Phụ lục: lệnh, artifact và code

### Lệnh chính

```bash
C=configs/confidence/kev.yaml
# 1. Cache trace cho các cohort phát triển
l2r4kie cache-traces --config $C --split train dev calibration risk_validation
# 2. Chọn head trên dev
l2r4kie select-heads --config $C
# 3. Calibration, ngưỡng, luật dạng (bundle đang serve)
l2r4kie finalize --config $C --set output=artifacts/kev-r4/confidence/selected-shapes --set shape_rules=true
# 4. Audit (chỉ sau khi finalize)
l2r4kie cache-traces --config $C --split audit
l2r4kie audit --config $C --set output=artifacts/kev-r4/confidence/selected-shapes
# 5. Infer
l2r4kie infer --confidence 'artifacts/kev-r4/confidence/selected-shapes/heads/attention[value@14+value@21]' ...

# Bản nhanh: thêm vào mọi bước từ 1, với cache và output riêng
#   --set merge_adapter=true [--set fast_decode=true]

# Đo tốc độ và độ khớp giữa hai cách decode
python scripts/compare_decode.py ...      # artifacts/kev-r4/profile/fast_vs_merged.json
python scripts/profile_infer.py ...       # tách thời gian prefill, decode, trace
python scripts/json_baseline.py ...       # baseline xuất JSON
# Audit2
python scripts/plan_fresh_audit.py --per-form 20 --seed 211 \
    --output artifacts/kev-r4/confidence/audit2/cohort_plan.json
```

Lệnh đầy đủ cho audit2 nằm trong `docs/results/fresh_audit.md`, cho đo tốc độ trong `docs/results/infer_speed.md`.

### Artifact

| Đường dẫn | Nội dung |
| --- | --- |
| `artifacts/kev-r4` | LoRA của extractor KevFormat |
| `artifacts/kev-r4/confidence/cache` | Trace của S (eager, chưa merge) |
| `artifacts/kev-r4/confidence/selected` | Head đã chọn, không luật dạng (R, ngưỡng 0,9362) |
| `artifacts/kev-r4/confidence/selected-shapes` | **Bundle đang serve** (S, ngưỡng 0,9220) |
| `artifacts/kev-r4/confidence/cache-merged`, `selected-merged` | Cache và bundle của M |
| `artifacts/kev-r4/confidence/cache-fast`, `selected-fast` | Cache và bundle của F |
| `artifacts/kev-r4/confidence/audit2/cohort_plan.json` | Cohort audit2 (318 tài liệu) |
| `artifacts/kev-r4/confidence/cache-audit2` | Trace audit2 của S |
| `artifacts/kev-r4/profile/fast_vs_merged.json` | Số đo 1,72× của fast_decode |

### File code chính

| File | Vai trò |
| --- | --- |
| `src/l2r4kie/model/format.py` | KevFormat: prompt, token đánh dấu, vị trí tín hiệu |
| `src/l2r4kie/model/decode.py` | Decode theo nhánh, dùng chung prefix ảnh, nén batch |
| `src/l2r4kie/model/early_stop.py` | Dừng sớm nhánh không còn thể 'ok' |
| `src/l2r4kie/model/fast_decode.py` | StaticCache + CUDA graph cho đuôi batch, `warm_up` |
| `src/l2r4kie/model/extractor.py` | Nạp model, `merge_adapter`, pass trace teacher-forced |
| `src/l2r4kie/confidence/features.py`, `heads.py` | Trích đặc trưng trạng thái ẩn và các kiểu head |
| `src/l2r4kie/confidence/calibration.py`, `policy.py` | Calibration, ngưỡng bảo thủ, lý do review |
| `src/l2r4kie/confidence/shapes.py` | Luật dạng theo field |
| `src/l2r4kie/confidence/bundle.py`, `cache.py` | Bundle serve, cache trace, kiểm tra serving settings |
| `src/l2r4kie/pipelines/` | `trace_cache`, `head_selection`, `finalize`, `audit`, `infer` |

### Tài liệu kết quả

- `docs/results/kev_extractor.md`: độ chính xác extractor.
- `docs/results/kev_confidence.md`: chọn head, calibration, cổng G1–G4.
- `docs/results/shape_rules.md`: luật dạng và ngưỡng fit lại.
- `docs/results/agreement_signal.md`: thí nghiệm tín hiệu đồng thuận (âm tính).
- `docs/results/infer_speed.md`: đo tốc độ và các tối ưu.
- `docs/results/fresh_audit.md`: đăng ký trước audit2.
