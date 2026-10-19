# Tốc độ infer sau khi train: đo được gì và tối ưu theo thứ tự nào

Trạng thái (2026-10-19): **đã đo; đã thêm `merge_adapter` và `fast_decode` (CUDA graph trong HF, mục 5), cả hai mặc định tắt; đang cache lại trace và audit trên cohort mới (`fresh_audit.md`).** Chưa làm vLLM. Mọi số đo đều trên H200 dùng chung với tiến trình khác, nên tuyệt đối có nhiễu. Tỷ lệ giữa các phần thì ổn định.

## 1. Thời gian đi đâu

`scripts/profile_infer.py`, 24 tài liệu dev (lấy cách đều theo id), **toàn bộ field** của mỗi tài liệu (667 field). Cấu hình giống lúc serve có bundle confidence: trace bật, layer 14 và 21, budget 16.384 token, early stop bật, 64 nhánh mỗi chunk, LoRA chưa merge. Kết quả ở `artifacts/kev-r4/profile/dev_trace.{jsonl,log}`.

| | Giá trị |
|---|---|
| Giây / tài liệu | trung vị **14,0**, trung bình 24,7, p90 59,5, tối đa 96,0 |
| Bước decode / tài liệu | trung bình 384, tối đa 1.202 |

| Phần | Tỷ lệ thời gian |
|---|---:|
| Đọc ảnh và processor (`encode_prefix`) | 2,7% |
| Prefill prefix (vision cộng text) | 0,9% |
| Prefill prompt của các nhánh | 0,6% |
| **Các bước decode (forward, đã sync GPU)** | **92,2%** |
| Host: lm_head, argmax, token stats, chép về CPU, vòng Python | 3,5% |
| *Trong đó: các bước chỉ còn ≤ 2 nhánh chạy* | ***80,1%*** |

**Kết luận 1: đuôi dài là vấn đề chính.** Ở 14/24 tài liệu, hơn nửa thời gian là lúc chỉ còn 1–2 field dài (thường là bảng hoặc array) được decode từng token. Mọi field khác đã xong từ lâu. Prefill, vision, chép KV của prefix đều không đáng kể.

**Kết luận 2: mỗi bước bị chặn bởi overhead, không phải compute.** Microbenchmark trên GPU 1 (tải 0%), KV 2.678 token, thời gian một bước:

| Cấu hình | 1 nhánh | 16 nhánh | 64 nhánh |
|---|---:|---:|---:|
| Hiện tại: LoRA chưa merge, eager | 73 ms | 74 ms | 77 ms |
| Đã merge LoRA, eager | 11–21 ms | 11 ms | 18 ms |
| Đã merge, StaticCache cộng `torch.compile(mode='reduce-overhead')` (CUDA graph) | **3,2 ms** | 6,9 ms | 19 ms |

- Đi từ 1 lên 64 nhánh gần như không tốn thêm thời gian. Model 2B chạy batch nhỏ chủ yếu tốn tiền launch kernel.
- LoRA chưa merge thêm 2 matmul nhỏ vào mỗi linear có LoRA, nên số kernel mỗi bước tăng mạnh.
- Compile lần đầu mất 14–25 s cho mỗi cỡ batch. Phải warm up khi khởi động server.

## 2. Merge LoRA: nhanh khoảng 2 lần, nhưng không giống hệt từng bit

Đã thêm `ExtractorConfig.merge_adapter`, mặc định `False`. Tùy chọn này gọi `merge_and_unload()` sau khi load adapter.

Kiểm tra trên đúng 24 tài liệu đó, 24 field đầu mỗi tài liệu (434 field, như trace cache dev), có trace và bundle `selected-shapes`. So với cache gốc (`artifacts/kev-r4/profile/merge_check.json`):

| | Chưa merge (đối chứng) | Đã merge |
|---|---:|---:|
| Value trùng cache | 98,4% | 97,5% |
| Status trùng cache | 99,3% | 99,5% |
| Lệch confidence: trung vị / p99 / max | 1e-6 / 0,017 / 0,050 | 7e-4 / 0,091 / **0,337** |
| Quyết định review bị đổi | 1 | **0** |
| Giây / tài liệu | 17,9 | **8,6** (nhanh 2,1×) |

- Đối chứng không trùng 100% với cache: cache được tạo trước khi có early stop, và bf16 cho kết quả hơi khác khi layout batch khác.
- Merge thêm khoảng 1 điểm value khác, và một số confidence lệch tới 0,34. Trên mẫu này không có quyết định review nào đổi, nhưng 434 field là quá ít để kết luận về policy.
- Nhanh 2,1× toàn trình, ít hơn mức 4–6× của từng bước, vì vòng Python của trace và các bước 64 nhánh không nhanh lên tương ứng.

**Quyết định:** chưa bật mặc định. Muốn serve bản đã merge kèm confidence thì phải cache lại trace bằng chính extractor đã merge, rồi chạy lại finalize (xem mục 4).

Đã làm để không trộn nhầm:
- `cache-traces` ghi `merge_adapter` / `fast_decode` vào metadata của cache.
- `finalize` từ chối các cache decode khác nhau, và ghi `serving` vào `bundle.json` và `frozen.json`.
- `infer --confidence` dùng đúng `serving` của bundle; yêu cầu một cờ mà bundle không có thì báo lỗi.
- `audit` từ chối trace audit decode khác với bundle.

## 3. Các phương án, xếp theo lợi ích chia cho chi phí

| # | Phương án | Lợi ích dự kiến | Chi phí, rủi ro | Phải validate lại? |
|---|---|---|---|---|
| 0 | **Luật dạng** (`shape_rules.md`) | review −2 điểm; không tốn GPU | đã xong | đã fit lại ngưỡng; cần audit mới |
| 1 | **Merge LoRA** | **2,1× toàn trình (đã đo)** | một cờ config; numerics bf16 đổi nhẹ | có: cache lại trace, chạy lại finalize |
| 2 | **CUDA graph cho bước decode trong HF**: StaticCache, batch theo bucket (1/2/4/…/64), compile khi warm up | đuôi dài: 15–21 ms xuống **3,2 ms mỗi bước** (thêm khoảng 5× so với 1). Ước tính toàn trình khoảng 1,5–3 s/tài liệu so với 14 s hiện nay | trung bình: thu gọn batch phải thành chuyển bucket (chép KV); StaticCache dài prefix + budget tốn bộ nhớ, cần bucket theo cả độ dài; vòng Python trace phải dồn vào buffer trên GPU, nếu không host sẽ thành nút cổ chai khi mỗi bước chỉ 3 ms | có (kernel fuse đổi numerics) |
| 3 | **Batch nhiều tài liệu cùng lúc** (throughput) | khi đuôi của tài liệu A chạy, field của các tài liệu khác lấp chỗ trống trong batch. 64 nhánh gần như cùng giá với 1 nhánh, nên throughput tăng **khoảng 5–15×** khi có hàng đợi | scheduler riêng trong HF (KV mỗi tài liệu một prefix), hoặc dùng vLLM | không, nếu numerics giữ nguyên |
| 4 | **vLLM sinh value, rồi một lượt HF teacher-forced lấy tín hiệu** | gộp 2 và 3: CUDA graph, paged KV, prefix caching (KV prefix không chép theo nhánh), continuous batching giữa các tài liệu | lớn: 1 request mỗi nhánh = prefix + prompt nhánh (vị trí M-RoPE trùng KevFormat, đã kiểm: token text sau ảnh tiếp từ max + 1); `allowed_token_ids` thay cho chặn special token; `stop_token_ids=[box_end]`; early stop cần logits processor riêng, hoặc chấp nhận budget. Tín hiệu (marker state, layer 14/21, token stats) lấy bằng **một forward teacher-forced kiểu `pack()`** trên value đã sinh (bằng decode trong FP32 theo `tests/test_decode.py`), rẻ vì chạy song song mọi token | có: token greedy của vLLM có thể khác HF |
| 5 | Chồng `encode_prefix` (CPU) lên decode của tài liệu trước | ≤ 2,7%; tăng lên khi decode nhanh hơn | thấp: thread hoặc process riêng | không |
| 6 | Đổi format cho bảng (chia mỗi dòng thành một nhánh) | rút chính đuôi dài: decode song song thay vì tuần tự | train lại extractor, chạy lại toàn bộ G1–G4 | có, toàn bộ |

Không đáng làm:
- Hạ `max_value_tokens`: đuôi ở mẫu này là value thật (đều kết thúc `close`, dài tới khoảng 1.200 token). Cắt budget là cắt bảng đúng.
- Tối ưu prefill hoặc tránh chép KV của prefix: tổng cộng chưa tới 2%.
- Sampling hay tín hiệu đồng thuận: âm tính, và còn tốn gấp đôi thời gian decode (`agreement_signal.md`).

## 4. Lộ trình đề xuất

1. **Ngay:** dùng `selected-shapes`, không đổi thời gian, giảm review.
2. **Bước 1 (khoảng 1 ngày):**
   - Thêm cờ `merge_adapter` vào config confidence. Ghi cờ này vào metadata của cache và `bundle.json`, và kiểm tra nó trong `check_source`.
   - Cache lại trace bằng extractor đã merge cho train, dev, calibration và risk_validation (1.533 tài liệu × khoảng 8 s, khoảng 3,5 giờ trên một GPU).
   - Giữ nguyên head đã chọn, chạy lại `finalize` (calibration cộng ngưỡng cộng luật dạng).
   - Nếu số trên dev và risk_validation không xấu đi so với `selected-shapes`, chuyển serving sang bản đã merge.
3. **Bước 2 (khoảng 1 tuần), chọn một trong hai:**
   - **Ở lại HF:** CUDA graph theo bucket, cộng buffer trace trên GPU, cộng scheduler nhiều tài liệu. Giữ được nguyên mã decode và kiểm soát numerics.
   - **Chuyển vLLM** (đề xuất, nếu cần throughput lớn): vLLM sinh value, rồi HF chạy `pack()` teacher-forced để lấy tín hiệu cho head. Gần như không phải tự viết scheduler hay CUDA graph.

   Cả hai hướng đều phải: cache lại trace bằng đường decode mới; so độ khớp value và confidence với bản HF (như mục 2); chạy lại finalize; và đo trên **cohort audit mới** với cổng non-inferiority về review và bắt lỗi so với bundle đang serve. Audit cũ đã dùng cho G4 nên không dùng lại.
4. **Chỉ khi bảng quan trọng:** đổi format bảng (mục 3, phương án 6). Cần train lại.

## 5. `fast_decode`: CUDA graph cho đuôi, rồi một lượt teacher-forced lấy tín hiệu

`model/fast_decode.py`, bật bằng `ExtractorConfig.fast_decode` (config confidence: `fast_decode: true`; cần `merge_adapter: true` để có lợi).

1. **Sinh token** (`generate`), cùng logits đã chặn special token, cùng greedy, cùng early stop như `decode_prefix`:
   - Lúc batch còn rộng: chạy eager như cũ, thu gọn batch khi số nhánh còn chạy giảm.
   - Khi còn ≤ 4 nhánh: chép KV sang `StaticCache` và chạy bước decode đã `torch.compile(mode='reduce-overhead')` (CUDA graph). Mỗi bucket (số dòng 1/2/4 × độ dài cache 4.096/8.192/16.384/32.768) có một graph riêng. Cache đầy thì chép sang bucket dài hơn.
   - Không chép gì về host ngoài token đã chọn.
2. **Lấy tín hiệu** (`trace_tokens`): một forward duy nhất trên prefix cache cho prompt cộng token đã sinh của mọi nhánh, gom các dòng có độ dài gần nhau (≤ 16.384 token mỗi lượt). Từ đó đọc state ở key/decide/value, layer 14/21, state sau từng token và thống kê token. Trong FP32, mọi thứ trùng với decode tăng dần (`tests/test_fast_decode.py`, 6 test).
3. **Warm up:** `fast_decode.warm_up(extractor, longest)` compile sẵn 12 graph. Trên H200 mất khoảng 300 s. Nếu không gọi thì graph được compile lần đầu dùng (khoảng 20 s mỗi graph) ngay trong request.

Lỗi đã gặp trên GPU thật và đã sửa: `StaticCache` cấp phát buffer ở lần update đầu. Nếu lần đó chạy trong hàm đã compile, buffer nằm trong vùng nhớ của CUDA graph và bị graph khác ghi đè. Bây giờ bước đầu chạy eager.

Kết quả đo: xem `fresh_audit.md` và `artifacts/kev-r4/profile/fast_vs_merged.json` (điền sau khi chạy xong).

## Chạy lại các số đo

```bash
# phân rã thời gian (GPU 0 dùng chung)
CUDA_VISIBLE_DEVICES=0 python scripts/profile_infer.py --split dev --documents 24 \
    --output artifacts/kev-r4/profile/dev_trace
```

Microbenchmark từng bước và kiểm tra merge là script tạm, không nằm trong repo. Cách đo đã mô tả ở mục 1 và mục 2.
