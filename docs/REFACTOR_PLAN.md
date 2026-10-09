# Kế hoạch refactor: l2r4kie cũ → repo này, kèm format Kev

Nguồn: `/home/jovyan/bachdx2/l2r4kie`, commit `0428235` (gọi là **cũ**). Đích: `src/l2r4kie/` (gọi là **mới**).

## Mục tiêu

Chuyển code sang repo mới và đổi format train/decode sang kiểu Kev:

```
input : <|vision_start|>[ảnh]<|vision_end|> prompt <|object_ref_start|>field_id: mô tả<|object_ref_end|><|box_start|>
output: value dạng text thuần<|box_end|>
```

Giống Kev, **không thêm token mới**. Format dùng lại các special token có sẵn nhưng ít dùng của Qwen2-VL (chi tiết ở mục "Chọn marker" bên dưới).

Các tín hiệu thu được trong cùng một lượt decode, không tốn thêm forward:

- `h_key`: hidden tại `<|object_ref_end|>`. Model đã đọc ảnh và field nhưng chưa sinh value. Đây là tương đương `h_decide`/query của Kev.
- `h_decide`: hidden tại `<|box_start|>`, vị trí cuối của prompt, chính là state sinh ra logits của token value đầu tiên.
- `h_value`: hidden tại `<|box_end|>`, sau khi value hoàn tất. Đây là tín hiệu correctness chính, tương đương closing marker của option trong Kev.
- Trace của từng token value, giữ như head attention hiện tại.

Value là text thuần nên không còn trạng thái `invalid_json`. Chỉ còn `ok` và `truncated` (sinh hết giới hạn token mà không gặp `<|box_end|>`).

### So với code cũ: điểm giữ và điểm đổi

| | Cũ | Mới |
|---|---|---|
| Cô lập field, prefix ảnh dùng chung, reset M-RoPE | có | **giữ nguyên** |
| Marker kết thúc value | `<\|im_end\|>`, dùng chung với chat template | `<\|box_end\|>`, chỉ dùng cho vai trò này |
| Value | JSON (`"abc"`, `[...]`, `true`) | text thuần |
| Tín hiệu trước khi sinh value | không có | `h_key`, `h_decide` |
| Vocab | không đổi | **không đổi** (dùng lại 4 special token có sẵn) |

Kỳ vọng phải đặt đúng mức:

- `h_end` cũ thực chất đã là hidden tại marker kết thúc value.
- Trên test của r4, lỗi JSON chỉ có 6/2150 field (0,3%). Lỗi bị cắt cụt là 47 (2,2%), chủ yếu ở mảng. Format mới không tự sửa được lỗi cắt cụt.

Lợi ích thật cần đo bằng thí nghiệm, gồm ba phần: marker đóng riêng (không dùng chung với chat template), tín hiệu `h_key`/`h_decide`, và text thuần. Mốc để so là r4 với head attention: EM test 89,0%, review 29,9% ở mức bắt ≥95% lỗi.

## Nguyên tắc

1. **Port trước, đổi format sau.** Phase A port code với format cũ (JSON + `<|im_end|>`) đặt sau một interface `ValueFormat`, và chứng minh ra đúng output như code cũ. Phase B chỉ thêm `KevFormat`. Khi đó mọi khác biệt về kết quả chỉ có thể đến từ format, không phải từ lỗi port. Phần decode/packing (KV cache, vị trí, cô lập nhánh) là chỗ dễ sai nhất nên cần mốc này.
2. **Mỗi step là một commit** trên branch `refactor/migrate`, `uv run pytest` phải xanh.
3. **Parity với code cũ.** Code cũ chạy bằng `/home/jovyan/bachdx2/l2r4kie/.venv/bin/python` và ghi output ra file (hai package trùng tên nên không import chung được). Fixture nhỏ để ở `tests/fixtures/`, script so sánh ở `scripts/parity/`.
4. **Mỗi module chỉ import module tầng dưới:** `utils` ← `data` ← `eval` ← `confidence` (phần thuần) ← `model` ← `train` ← pipeline confidence ← `cli`.
5. **Có cổng quyết định.** Phase C có tiêu chí go/no-go định trước. Format Kev không thắng thì giữ format cũ, không ép.

## Chọn marker: dùng lại token ít dùng như Kev

Kev ([kev/model.py](https://github.com/jaredpalmer/kev/blob/main/kev/model.py)) không thêm token. Nó dùng lại 5 special token ít dùng của Qwen2.5 (`SPECIAL = ["<|fim_prefix|>", "<|fim_middle|>", "<|box_start|>", "<|box_end|>", "<|fim_suffix|>"]`) cho state, question, mở option, đóng option và decide. Embedding của các token này chỉ được train thêm khi bật tùy chọn (`trainable_token_indices`). Text đầu vào được escape mẫu `<|name|>` để người dùng không tạo ra được token điều khiển.

Qwen2-VL-2B **không có** `<|fim_*|>`: chỉ có 14 special token. Đã đo embedding trên model thật (hàng dự trữ chưa train có norm 0,435; token thường có norm trung vị 0,600):

| Token | Norm | Cos với hàng chưa train | Đánh giá |
|---|---:|---:|---|
| `object_ref_start` / `object_ref_end` | 0,574 / 0,551 | 0,45 / 0,35 | Đã train rõ ràng. **Dùng cho key.** |
| `box_start` / `box_end` | 0,483 / 0,521 | 0,63 / 0,72 | Đã train. **Dùng cho value**, giống cặp option của Kev. |
| `quad_start` / `quad_end` | 0,442 / 0,501 | 0,77 / 0,63 | Train yếu. Dự phòng. |
| `vision_start/end`, `vision_pad`, `image_pad`, `video_pad` | 0,435 | 1,00 | Trùng hàng chưa train, và `get_rope_index` dùng các token này để định vị ảnh. **Không được dùng.** |
| `im_start`, `im_end`, `endoftext` | | | Thuộc chat template và padding. Không dùng. |

Probe zero-shot trên base model, 24 field của 6 tài liệu test ([scripts/probe_markers.py](../scripts/probe_markers.py)):

| Cặp value | Đóng đúng marker | Sinh tọa độ thay vì value | Special token lạc |
|---|---|---|---|
| `object_ref` + `box` | **24/24** | 24/24 | 0 |
| `object_ref` + `quad` | 0/24 (đóng bằng `box_end`) | 24/24 | 24/24 |

Kết luận:

- `box_end` là marker đóng mà base model đã sinh rất ổn định. Điều này giảm lỗi format (không đóng được, cắt cụt). Cặp `quad` thì base model không biết đóng.
- Cái giá phải trả: Qwen2-VL được pretrain grounding theo mẫu `<|object_ref_start|>tên<|object_ref_end|><|box_start|>(x1,y1),(x2,y2)<|box_end|>`, nên ban đầu model sẽ sinh tọa độ. Fine-tune phải đổi nội dung từ tọa độ sang text. Đây là thay đổi nội dung chứ không phải cấu trúc, nhưng cần đo: đếm output khớp regex tọa độ như một loại lỗi riêng, và theo dõi loss ở các bước đầu.
- Hai ý phụ đáng ghi lại. Một: chưa train mà model đã khoanh được vùng gần đúng cho một số field, ví dụ "Mẫu số" → `(684,10),(996,101)`. Grounded extraction (sinh box rồi mới sinh value) có thể dùng đúng prior này, xem đề xuất E cũ. Hai: nếu sau này đổi sang Qwen2.5-VL, có thể dùng nguyên bộ token `fim` như Kev.
- Không cần marker riêng cho tài liệu: ảnh đã được bọc sẵn bằng `<|vision_start|>…<|vision_end|>`. Cũng không cần `</end>`: `box_end` vừa để dừng vừa làm tín hiệu.

Chống lỗi format, theo thứ tự:

1. Escape mẫu `<|name|>` trong field id, mô tả và value khi tokenize, như Kev. Dữ liệu hiện tại không có mẫu này, nhưng request lúc inference có thể có.
2. Khi decode value, chặn mọi special token trừ `box_end` bằng logit mask. Khi đó lỗi format chỉ còn một loại: cắt cụt. Thống kê logprob/entropy tính trên logits đã mask, và train cache với serving đi cùng một đường decode.
3. Đo riêng: tỷ lệ cắt cụt, tỷ lệ output dạng tọa độ.

## Quyết định cần chốt

| # | Câu hỏi | Đề xuất |
|---|---|---|
| K1 | Marker | `object_ref` cho key, `box` cho value (bảng trên). Không thêm token, không dùng `</end>`. Dự phòng: giữ `<\|im_end\|>` làm marker đóng nếu prior tọa độ không gỡ được. |
| K2 | Mảng object (14.760 field, 4,6%) xử lý thế nào? | Giai đoạn đầu: đặt JSON text vào trong value, metric báo riêng. Tách thành từng ô cần row detector, để sau. |
| K3 | Bool biểu diễn thế nào? | Text `true`/`false`. |
| K4 | Có giữ chat template quanh prefix ảnh không? | Giữ system/user để tận dụng instruct prior; chỉ phần branch dùng marker. Format được thiết kế để cấu hình được. |
| K5 | Đi qua Phase A với format cũ để kiểm parity? | Có, theo Nguyên tắc 1. |
| K6 | Có train embedding của 4 marker không? | Mặc định không (các token đã có embedding đã train). Bật `trainable_token_indices` như một ablation, giống tùy chọn `special_embeddings` của Kev. |
| D1 | Phạm vi port | Chỉ pipeline G3. Bỏ joint loss, synthetic negatives, mining, head `feature` (danh sách ở cuối file). |
| D2 | Artifacts | Không copy. Config trỏ tới repo cũ. |

---

# Phase A: Port, giữ format cũ, có parity

## A0: Chuẩn bị

- Thêm dependency theo bản pin cũ: `torch==2.12.0`, `torchvision==0.27.0`, `transformers==4.57.6`, `peft==0.18.1`, `accelerate>=1.15`, `numpy>=2.4`, `pillow>=12.2`, `pyyaml>=6.0`, extra `report = [matplotlib>=3.10]`.
- Tạo `scripts/parity/dump_old.py` để xuất golden output từ code cũ.

## A1: `utils/`

| Cũ | Mới |
|---|---|
| `data.read_jsonl`, `write_json`; mẫu ghi `.tmp` rồi `replace` lặp ở 3 chỗ | `utils/io.py` |
| `data.checkpoint_fingerprint` | `utils/fingerprint.py` |
| `application.load_config` | `utils/config.py` |

- **Parity:** fingerprint của checkpoint r4 khớp `source_fingerprint` trong `token-review-v2/selected/frozen_selection.json`.

## A2: `data/`

| Cũ | Mới |
|---|---|
| `domain.escape`, `branches` | `data/schema.py` |
| `domain.Claim` | `data/types.py`: `FieldRequest(field_id, description)` cho inference, `Claim(field_id, description, value)` cho train |
| `data.prepare` | `data/prepare.py`: tách `group_duplicate_pages`, `assign_split`, `prepare` |
| `application.select_documents` | `data/selection.py` |
| `cache_token_review.plan_cohorts` | `data/cohorts.py` |

- **Sửa lỗi #1 (leakage):** danh sách file loại trừ phải truyền tường minh, thiếu file thì raise lỗi, bỏ đường dẫn hardcode theo CWD.
- **Parity:** `select_documents` ra cùng danh sách ID trên mọi split. `plan_cohorts` tái tạo đúng `token-review-winner/cache/cohort_plan.json`.

## A3: `eval/`

| Cũ | Mới |
|---|---|
| `domain.canonical`, `correct` | `eval/comparator.py` |
| `application.metrics` | `eval/metrics.py::confidence_metrics` |
| `review.evaluate_policy`, `review_curve`, `budgets`, `wilson`, `bootstrap_comparison`, `within_field_metrics` | `eval/review_metrics.py` |

- **Sửa lỗi #5:** metrics không còn kéo theo transformers. **Sửa lỗi #3:** báo thêm `head_error_recall`, không tính các field bắt buộc review.
- **Parity:** tính lại từ `audit_predictions.jsonl` phải khớp `audit_report.json`.

## A4: `confidence/` phần thuần

| Cũ | Mới |
|---|---|
| `token_confidence.*` | `confidence/features.py` |
| `TokenConfidenceHead`, `make_head` | `confidence/heads.py` (kind `token` và `linear`) |
| `fit_calibration`, `select_calibration` | `confidence/calibration.py` |
| `ReviewPolicy`, `fit_policy`, `fit_conservative_policy`, `queue_fields` | `confidence/policy.py` |

- Phần phụ thuộc format trong `build_trace` (mask token cấu trúc JSON, one-hot 6 kiểu JSON) đưa vào hook của `ValueFormat`.
- **Parity:** head attention cũ chấm `audit.pt` cho Δlogit = 0; calibration khớp; ngưỡng ra 0,9597.

## A5: `model/`, có interface `ValueFormat`

| Cũ | Mới |
|---|---|
| system prompt, `branch_prompt`, `<\|im_end\|>`, `json.dumps`/`json.loads` value | `model/formats.py`: `ValueFormat` (xem dưới) và `JsonImEndFormat` |
| `packing.block_mask`, `Packed`, `pack` | `model/packing.py` (nhận format) |
| `Extractor` (load, save, `shared_inputs`) | `model/extractor.py` (device lấy từ config) |
| `extract`, `_decode` | `model/decode.py`: tách vòng decode khỏi bước chấm điểm |

```python
class ValueFormat(Protocol):
    marker_ids: dict[str, int]                # vai trò → id token có sẵn; rỗng với format cũ
    def prefix_messages(self, images) -> ...  # phần tài liệu dùng chung
    def branch_prompt_ids(self, tok, field) -> list[int]   # kết thúc ngay trước value
    def target_ids(self, tok, value) -> list[int]          # value + marker kết thúc
    stop_id: int                               # marker kết thúc value
    signal_offsets: dict[str, int]             # h_key/h_decide: vị trí tính từ cuối prompt
    banned_ids: list[int]                      # special token bị mask khi decode value
    def parse(self, text) -> tuple[value, status]
    def content_mask(self, tok, ids) -> list[bool]
    def kind_features(self, value) -> list[float]
```

- **Parity (GPU):** 5 tài liệu audit, checkpoint attention, cùng precision: value và status giống hệt, Δconfidence ≤ 1e-6, chạy thêm một lần ở FP32. Port `scripts/audit_isolation.py`.

## A6: `train/`

| Cũ | Mới |
|---|---|
| `validate_config` cộng các `config.get` rải rác | `train/config.py::TrainConfig` |
| `Extractor.losses` | `train/losses.py` (chỉ CE) |
| `lr_multiplier` | `train/schedule.py` |
| `latest_snapshot`, `save_snapshot`, `restore_state` | `train/snapshots.py` |
| `application.train`, `usable` | `train/trainer.py` |

- **Sửa lỗi #4:** head không nằm trong optimizer khi train extractor.
- **Parity (GPU):** 20 step config r4, cùng seed: loss và `document_id` từng step khớp code cũ; resume ở step 10 cho kết quả như chạy liền.

**Mốc Phase A:** code mới tái tạo được extractor r4 và kết quả confidence cũ.

---

# Phase B: Format Kev

## B1: Value dạng text (`data/serialize.py`, `eval/comparator.py`)

- `to_text(value)`: string giữ nguyên (kể cả xuống dòng, số 0 đầu); `""` thành value rỗng; bool thành `true`/`false`; mảng object theo K2.
- `text_correct(pred, target)`: NFC + bỏ khoảng trắng hai đầu, giữ hoa/thường. Comparator cũ không đổi.
- **Test:** chuyển hai chiều cho mọi kiểu value có trong data. 15.978 value số có 0 đầu giữ nguyên.
- **Đo:** chấm lại prediction của r4 bằng comparator text, báo mức chênh với comparator JSON, để Phase C so cùng một thước.

## B2: Marker có sẵn, escape và mask

- `model/markers.py`: tra id theo tên (`convert_tokens_to_ids`), assert đúng 4 id `object_ref_start/end`, `box_start/end` và assert chúng không trùng token ảnh. Tokenizer và vocab giữ nguyên.
- `escape_specials(text)`: đổi mẫu `<|name|>` trong field id, mô tả và value trước khi tokenize, như Kev.
- `banned_ids`: mọi special id trừ `box_end`, mask khi decode value.
- Tùy chọn (K6, mặc định tắt): train embedding của 4 marker bằng `LoraConfig(trainable_token_indices={'embed_tokens': ids})`. Vì embedding và LM head dùng chung trọng số, cần kiểm tra PEFT 0.18.1 cập nhật cả hai phía nhất quán.
- **Test:** tokenize một mô tả chứa `<|box_end|>` không sinh ra id marker; decode có mask không bao giờ ra special token khác `box_end`; khi bật K6 thì sau một bước optimizer chỉ 4 hàng marker thay đổi.

## B3: `KevFormat` (`model/formats.py`)

- Prefix là chat template system/user bọc ảnh (theo K4). Branch là `<|object_ref_start|>field_id: mô tả<|object_ref_end|><|box_start|>`. Target là text + `<|box_end|>`.
- Packed train và cached decode dùng chung một object format. Decode dừng ở `<|box_end|>`, có mask `banned_ids`.
- Lấy `h_key` tại `<|object_ref_end|>` và `h_decide` tại `<|box_start|>` (cả hai nằm trong prompt, không tốn thêm forward), `h_value` tại `<|box_end|>`.
- **Theo dõi khi train:** tỷ lệ output dạng tọa độ trên dev ở các mốc 50/100/200 bước. Không giảm về ~0 thì chuyển marker đóng sang dự phòng của K1.
- **Test:** chạy toàn bộ test cô lập cho cả hai format. `h_value` ở cached decode khớp với teacher forcing qua packed train (sai số FP32 ≤ 1e-3, như audit cũ). Đổi `h_key` của một nhánh không ảnh hưởng nhánh khác.

## B4: Feature và head cho format mới

- `content_mask` đơn giản đi vì không còn token cấu trúc JSON. `kind_features` thành rỗng/bool/text/mảng, nên `SUMMARY_SIZE` thành tham số.
- Head nhận thêm `h_key` (tùy chọn) và có mode `query` để ablation: so `h_key` với `h_value` và trace.
- Head và calibration cũ không dùng được với format mới, phải train lại toàn bộ (Phase C).

---

# Phase C: Thí nghiệm có cổng quyết định (GPU)

## C0: Khai báo cohort trước khi chạy

- Dùng lại dev 116 và test 116 tài liệu của sweep để so extractor trên cùng tài liệu.
- Cho phần confidence: tài liệu chưa dùng của 13 loại hiếm **đã hết**. Phải chọn trước một trong hai: (a) chỉ audit trên các loại còn tài liệu mới; (b) dùng lại audit cũ và ghi là "đã được dự án xem". Cohort plan ghi ra trước khi decode.

## C1: Train extractor format Kev

- Siêu tham số như r4: 2,1 MP, 12 field, CE-only, 1000 step, seed 42, LoRA r16, marker theo K1, K6 tắt. Ablation phụ: bật K6.
- **Cổng:** EM dev (comparator text, cả hai model chấm cùng một thước) không thấp hơn r4 quá biên định trước, tính bằng paired bootstrap theo tài liệu. Báo riêng mảng object và tỷ lệ cắt cụt.

## C2: Confidence trên extractor mới

- Chạy pipeline G3 (cache trace → chọn head trên dev → calibration → ngưỡng trên risk → audit).
- Ablation trên cùng trace: chỉ `h_value`; `h_value` + trace (attention); thêm `h_key`.
- **Cổng:** review rate ở mức bắt ≥95% lỗi so với 29,9% của r4 + attention. Tách phần đóng góp của extractor (giới hạn review lý thuyết thay đổi theo tỷ lệ lỗi) khỏi đóng góp của head.

## C3: Kết luận

- Kev thắng: đặt `KevFormat` làm mặc định, xóa `JsonImEndFormat` và checkpoint cũ khỏi config.
- Kev thua hoặc hòa: giữ format cũ, ghi kết quả vào `docs/history.md`.

---

# Phase D: CLI, configs, tài liệu

- `cli.py` có các lệnh `prepare`, `train`, `evaluate`, `cache-traces`, `select-heads`, `finalize`, `audit`, `infer`. Response của `infer` bỏ trạng thái `invalid_json` nếu dùng Kev.
- `configs/extractor/{r4_json,kev}.yaml`, `configs/extractor/experiments/`, `configs/confidence.yaml`.
- README; `docs/results/`; `docs/history.md` ghi G1/G2, đối chiếu Kev và kết quả Phase C. Xóa `scripts/parity/`.

## Phần không port (theo D1)

- **Code:** `application.predictions(calibrate=True, mine=True)`, `domain.negative`, `FeatureHead`, phần riêng của G2 trong `review_optimization`, candidate "head có sẵn".
- **Scripts:** `audit_visual_confidence`, `cache_review_audit`, `cache_review_features`, `finalize_review_policy`, `optimize_review`, `replay_review_serving`, `plot_review_report`, `train_token_review`, `run_token_review_experiment`, `run_confidence_on_winner`, `rerun_confidence_v2`, `finish_training_optimization`, `probe_training_attention`.
- **Configs:** `pilot`, `refine`, `mixed-smoke`, `full`.

## Theo dõi

- [ ] A0 Chuẩn bị · [ ] A1 utils · [ ] A2 data · [ ] A3 eval · [ ] A4 confidence (thuần) · [ ] A5 model + ValueFormat · [ ] A6 train
- [ ] B1 serialize · [ ] B2 marker/escape/mask · [ ] B3 KevFormat · [ ] B4 feature/head
- [ ] C0 cohorts · [ ] C1 extractor Kev · [ ] C2 confidence · [ ] C3 kết luận
- [ ] D CLI, configs, tài liệu
