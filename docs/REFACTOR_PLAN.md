# Kế hoạch refactor: chuyển và sửa từng phần

- **Nguồn (cũ):** `/home/jovyan/bachdx2/l2r4kie`, commit `0428235`.
- **Đích (mới):** `src/l2r4kie/` trong repo này.
- **Bằng chứng chọn marker:** [notes/marker_selection.md](notes/marker_selection.md).

## Cách làm

Mỗi step chuyển **một phần của pipeline** và **sửa luôn** trong step đó, kể cả đổi sang KevFormat. Thứ tự đi theo đường dữ liệu chạy qua hệ thống:

```
1. Dữ liệu  →  2. Tiền xử lý (input model)  →  3. Infer  →  4. Train  →  5. Eval  →  6. Confidence  →  7. CLI + tài liệu
```

Quy tắc cho mọi step:

- **Một step là một PR/commit**, bạn review xong mới sang step sau.
- **CLI lớn dần theo step.** Step nào xong thì thêm lệnh của step đó vào `l2r4kie`, để bạn chạy thử và nhìn output thật.
- Mỗi step có mục **"Bạn kiểm tra"**: lệnh chạy được ngay và output mong đợi.
- **So với code cũ** chỉ ở những phần không bị đổi format: dữ liệu, metrics, calibration, policy. Phần infer và train đổi format nên được kiểm bằng test cấu trúc (tiny model), smoke test, và kết quả thí nghiệm so với r4.
- Code cũ và các artifacts của nó (r4, head attention) giữ nguyên, làm **mốc so sánh** ở step 5 và 6.

### Các quyết định đã chốt

| # | Quyết định |
|---|---|
| K1 | Marker có sẵn, không thêm token: `<\|object_ref_start\|>`/`<\|object_ref_end\|>` cho key, `<\|box_start\|>`/`<\|box_end\|>` cho value. Dự phòng: dùng `<\|im_end\|>` làm marker đóng nếu model tiếp tục sinh tọa độ. |
| K2 | Mảng object (4,6% field): JSON compact đặt trong value; metric báo riêng. |
| K3 | Bool → `true`/`false`. String rỗng → value rỗng. Data không có null hay số. |
| K4 | Prefix ảnh giữ chat template system/user; chỉ phần branch dùng marker. |
| K6 | Mặc định không train embedding của marker. `trainable_token_indices` để làm ablation. |
| D1 | Chỉ chuyển pipeline cuối cùng (G3): extractor CE-only + confidence head trên trace. Không chuyển joint loss, synthetic negatives, mining, `FeatureHead` (mục "Không chuyển" ở cuối). |
| D2 | Không copy artifacts; config trỏ tới dữ liệu và checkpoint ở repo cũ. |

### Format đích

```
prefix (dùng chung, encode 1 lần): system + user[ <|vision_start|>ảnh<|vision_end|> ... ]
branch (mỗi field, cô lập):        <|object_ref_start|>field_id: mô tả<|object_ref_end|><|box_start|>
output:                            value dạng text<|box_end|>
tín hiệu:                          h_key @ object_ref_end · h_decide @ box_start · h_value @ box_end
```

---

## Step 0: Nền móng

**Chuyển và sửa**

| Từ (cũ) | Sang (mới) | Sửa gì |
|---|---|---|
| `data.read_jsonl`, `write_json`; 3 chỗ ghi `.tmp` rồi `replace` | `utils/io.py` | Gom về một bộ hàm atomic dùng chung |
| `data.checkpoint_fingerprint` | `utils/fingerprint.py` | Giữ nguyên thuật toán |
| `application.load_config` | `utils/config.py` | Thêm `--set key=value` để ghi đè từ CLI |
| (rải rác) | `utils/seed.py` | `seed_everything` |

Ngoài ra:
- Thêm dependency theo bản pin cũ (`torch==2.12.0`, `torchvision==0.27.0`, `transformers==4.57.6`, `peft==0.18.1`, `accelerate`, `numpy`, `pillow`, `pyyaml`).
- Marker `gpu` khai báo trong `pyproject.toml`; `tests/conftest.py` tự bỏ qua test GPU khi không có CUDA.
- `cli.py` có khung `argparse` và lệnh `fingerprint`; `[project.scripts] l2r4kie`; `__main__.py`; xóa `main.py`.

**Kết quả:** `uv run l2r4kie --help` chạy được; `utils/` có test.

**Bạn kiểm tra**
```bash
uv run pytest -q
uv run l2r4kie fingerprint /home/jovyan/bachdx2/l2r4kie/artifacts/training-optimization/r4_2m_12f
# phải bằng "source_fingerprint" trong artifacts/token-review-v2/selected/frozen_selection.json của repo cũ
```

---

## Step 1: Dữ liệu (đọc dataset gốc → các split)

**Chuyển và sửa**

| Từ (cũ) | Sang (mới) | Sửa gì |
|---|---|---|
| `domain.escape`, `domain.branches` | `data/schema.py` | Đổi tên `escape_pointer`, `iter_branches`; trả `FieldSpec` thay vì dict |
| (dict tự do) | `data/types.py` | `Document`, `FieldSpec`, `FieldRequest` (inference), `Claim` (train). Bỏ kiểu hack `Claim(id, desc, None, 0)` |
| `data.prepare` (một hàm dài) | `data/prepare.py` | Tách thành `hash_pages`, `group_duplicates` (union-find), `assign_split`, `load_fields`, `prepare`. Định dạng output không đổi |
| `application.select_documents` | `data/selection.py` | Tham số tường minh (`seed`, `forms`, `holdout_percent`, `balanced_forms`, `limit`) thay vì cả dict config |
| `scripts/cache_token_review.py::plan_cohorts` | `data/cohorts.py` | **Sửa lỗi leakage:** danh sách loại trừ truyền tường minh, thiếu file thì báo lỗi, bỏ đường dẫn hardcode theo thư mục chạy |

**Kết quả**
- CLI: `l2r4kie prepare --data <gốc> --output <dir>`, `l2r4kie data-stats --prepared <dir>`, `l2r4kie plan-cohorts --config <yaml>`.
- Test: `tests/test_data.py`, `tests/test_cohorts.py`.

**Bạn kiểm tra**
```bash
uv run l2r4kie prepare --data /home/jovyan/bachdx2/data/kie-all-v1 --output artifacts/data
diff <(sort artifacts/data/test.jsonl) <(sort /home/jovyan/bachdx2/l2r4kie/artifacts/data/test.jsonl) && echo "test split giống hệt"
# làm tương tự cho train/dev/calibration và report.json (trừ trường đường dẫn source)
uv run l2r4kie data-stats --prepared artifacts/data
# in: số tài liệu mỗi split (9.535 / 870 / 860 / 728), 58 loại, số field theo kiểu value
```

**Xong khi**
- 4 split và `report.json` giống hệt bản cũ.
- `plan-cohorts` với cấu hình của lượt v1 tái tạo đúng cohort trong `token-review-winner/cache/cohort_plan.json`.
- Test "thiếu file loại trừ thì báo lỗi" xanh.

---

## Step 2: Tiền xử lý (tài liệu → input của model, theo KevFormat)

Đây là step đổi format. Mọi thứ biến một tài liệu thành token đều nằm ở đây.

**Chuyển và sửa**

| Từ (cũ) | Sang (mới) | Sửa gì |
|---|---|---|
| `json.dumps(value)` trong `packing` | `data/serialize.py` | **Mới:** `to_text`/`from_text`. String giữ nguyên (xuống dòng, số 0 đầu), rỗng thì rỗng, bool thành `true`/`false`, mảng object thành JSON compact |
| (không có) | `model/markers.py` | **Mới:** tra id 4 marker có sẵn (assert không trùng token ảnh); `escape_specials` (đổi `<\|x\|>` thành `<¦x¦>` như Kev); `banned_ids` (mọi special trừ `box_end`) |
| system prompt trong `Extractor.shared_inputs`, `packing.branch_prompt`, end `<\|im_end\|>` | `model/format.py` | **Thay bằng KevFormat:** `prefix_messages`, `branch_prompt_ids` (`object_ref` … `box_start`), `target_ids` (text + `box_end`), `signal_offsets` (`h_key=-2`, `h_decide=-1`), `parse`, `content_mask`, `kind_vector` |
| `packing.block_mask`, `pack` | `model/packing.py` | Nhận format; lưu thêm vị trí `h_key`/`h_decide`/`h_value` trong batch; encode value qua một hàm duy nhất (sửa chỗ `add_special_tokens` dùng không thống nhất) |

**Kết quả**
- CLI: `l2r4kie show-input --prepared artifacts/data --doc <id> [--fields 3]` in chuỗi token của prefix và từng nhánh, có tô marker, kèm vị trí của 3 tín hiệu.
- Test: `tests/test_serialize.py`, `tests/test_markers.py`, `tests/test_packing.py` (block mask; tiny Qwen2-VL: đổi value nhánh A thì nhánh B Δ = 0; đổi thứ tự nhánh không đổi kết quả).

**Bạn kiểm tra**
```bash
uv run l2r4kie show-input --prepared artifacts/data --doc <một id bất kỳ> --fields 3
# nhìn thấy:  ...<|object_ref_start|>/Họ tên: Họ và tên…<|object_ref_end|><|box_start|>NGUYỄN VĂN A<|box_end|>
#            h_key @ pos …, h_decide @ pos …, h_value @ pos …
uv run pytest tests/test_serialize.py tests/test_markers.py tests/test_packing.py -q
```

**Xong khi**
- Chuyển hai chiều text ↔ value đúng cho mọi kiểu value trong data.
- Mô tả chứa `<|box_end|>` không tạo ra id marker.
- Test cô lập nhánh xanh.

---

## Step 3: Infer (load model → decode → response JSON)

**Chuyển và sửa**

| Từ (cũ) | Sang (mới) | Sửa gì |
|---|---|---|
| `Extractor.__init__`, `save`, các property | `model/extractor.py` | Device lấy từ config (bỏ `cuda:1`); extractor không còn giữ head/calibration (chuyển sang step 6) |
| `Extractor.extract`, `_decode` | `model/decode.py` | Giữ cơ chế cũ (prefix encode 1 lần, nhân KV, decode các nhánh song song, chia chunk `max_branches`). **Thêm:** mask `banned_ids` trước argmax; thu `h_key`/`h_decide` từ prefill nhánh và `h_value` tại `box_end`; trả `DecodeResult(field_id, value, raw, status, signals, trace)`. Status chỉ còn `ok`, `truncated` (và `invalid_array` cho mảng) |
| phần `infer` trong `cli.py` | `pipelines/infer.py` | Response giữ schema cũ (`result`, `confidence`, `status`, `review`…); `confidence` là `null` cho tới step 6 |
| `scripts/audit_isolation.py` | `scripts/audit_isolation.py` | Chuyển sang API mới |

**Kết quả**
- CLI: `l2r4kie infer --model Qwen/Qwen2-VL-2B-Instruct [--adapter <dir>] --request req.json --output resp.json`.
- Test: `tests/test_decode.py` (tiny model): decode dừng ở `box_end`; không sinh token bị cấm; `h_value` khi decode khớp `h_value` khi teacher forcing qua packing (FP32, Δ ≤ 1e-5); vision forward đúng 1 lần.

**Bạn kiểm tra**
```bash
uv run l2r4kie infer --model Qwen/Qwen2-VL-2B-Instruct --request examples/request.json --output /tmp/resp.json
cat /tmp/resp.json
# model CHƯA train: value thường là tọa độ "(684,10),(996,101)". Đây là kết quả đúng mong đợi
# (prior grounding, xem notes/marker_selection.md). Điều cần kiểm ở đây là cấu trúc:
# mọi field có status, không có special token lạc, mỗi tài liệu 1 lần vision forward.
uv run python scripts/audit_isolation.py --model Qwen/Qwen2-VL-2B-Instruct --doc <id>
```

**Xong khi:** test decode xanh; `infer` chạy được trên GPU với request nhiều trang và nhiều field; audit isolation báo hidden của nhánh khác Δ = 0.

---

## Step 4: Train extractor

**Chuyển và sửa**

| Từ (cũ) | Sang (mới) | Sửa gì |
|---|---|---|
| `training.validate_config` cộng các `config.get` rải rác | `train/config.py` | Dataclass `TrainConfig`; khóa lạ trong yaml thì báo lỗi; thêm `format`, `trainable_markers` (K6) |
| `Extractor.losses` | `train/losses.py` | Chỉ còn CE trên value + `box_end`. Bỏ BCE/ranking/negatives |
| `training.lr_multiplier` | `train/schedule.py` | Giữ nguyên |
| `training.*_snapshot`, `restore_state` | `train/snapshots.py` | Giữ cơ chế an toàn (con trỏ chỉ đổi sau khi snapshot xong, giữ 2 bản) |
| `application.train` | `train/trainer.py` | Tách `claims_for(step)`, `train_step`, `maybe_snapshot`, `summary`. **Sửa:** optimizer không nhận head (trước đây head bị weight decay vô ích); bỏ lời gọi `negative()` làm tiêu RNG. **Thêm:** callback mỗi N step decode vài tài liệu dev và log EM, tỷ lệ output tọa độ, tỷ lệ cắt cụt |
| `configs/experiments/r4_2m_12f.yaml` | `configs/extractor/kev_smoke.yaml`, `kev_r4.yaml` | Cấu hình smoke (8 tài liệu, 30 step) và cấu hình thật giống r4 (2,1 MP, 12 field, 1000 step, seed 42) |

**Kết quả**
- CLI: `l2r4kie train --config configs/extractor/kev_smoke.yaml [--set steps=50]`.
- Test: `tests/test_training.py` (resume khôi phục optimizer/scheduler/RNG; accumulation lẻ; khóa config lạ; khi bật K6 chỉ 4 hàng embedding marker thay đổi).

**Bạn kiểm tra**
```bash
uv run l2r4kie train --config configs/extractor/kev_smoke.yaml
tail -3 artifacts/kev-smoke/train.jsonl
# loss giảm rõ; log callback: coordinate_rate giảm từ ~1.0 về gần 0 trên các tài liệu đã thấy
uv run l2r4kie infer --model Qwen/Qwen2-VL-2B-Instruct --adapter artifacts/kev-smoke --request examples/request.json --output /tmp/resp.json
# value giờ là text, không còn tọa độ
```

**Xong khi**
- Smoke chạy hết, loss hữu hạn và giảm, tỷ lệ tọa độ giảm.
- Dừng giữa chừng rồi resume cho kết quả như chạy liền.

**Chạy thật (sau khi bạn duyệt smoke):** `l2r4kie train --config configs/extractor/kev_r4.yaml`, khoảng 1,5 giờ trên H200. Theo dõi tỷ lệ tọa độ ở step 50/100/200/500. Nếu ở step 500 vẫn chưa về gần 0, chạy phương án dự phòng `--set format.close=im_end`.

---

## Step 5: Eval extractor

**Chuyển và sửa**

| Từ (cũ) | Sang (mới) | Sửa gì |
|---|---|---|
| `domain.canonical`, `correct` | `eval/comparator.py` | Thêm `text_correct` (NFC + strip, giữ hoa/thường; mảng so sau khi chuẩn hóa JSON). Giữ `json_correct` để chấm lại r4 |
| `application.metrics` | `eval/metrics.py` | **Sửa:** không còn kéo transformers chỉ để tính số |
| `error_kind`, bảng metrics trong `scripts/report_training_optimization.py` | `eval/errors.py`, `eval/extraction.py` | Thêm loại lỗi `coordinates`; EM micro/macro theo form, EM mảng, cắt cụt |
| `application.predictions` (chế độ evaluate) | `pipelines/predict.py` | Bỏ chế độ mine/calibrate |
| bootstrap trong report script | `eval/compare.py` | Paired bootstrap theo tài liệu giữa hai bộ prediction |

**Kết quả**
- CLI: `l2r4kie evaluate --config <yaml> --adapter <dir> --split dev --output <dir>`; `l2r4kie compare --a <pred.jsonl> --b <pred.jsonl>`.
- Test: `tests/test_metrics.py`, `tests/test_comparator.py`.
- Báo cáo `docs/results/kev_extractor.md`.

**Bạn kiểm tra**
```bash
# 1. Metrics mới khớp số cũ: chấm lại prediction dev của r4 bằng json_correct
uv run l2r4kie evaluate-file --predictions /home/jovyan/bachdx2/l2r4kie/artifacts/training-optimization/r4_2m_12f/dev.jsonl --comparator json
# phải ra EM 87,00% như TRAINING_OPTIMIZATION_REPORT.md
# 2. Kev so với r4 trên cùng 116 tài liệu dev, cùng comparator text
uv run l2r4kie evaluate --config configs/extractor/kev_r4.yaml --adapter artifacts/kev-r4 --split dev --output artifacts/kev-r4/eval-dev
uv run l2r4kie compare --a <r4 dev, chấm bằng text> --b artifacts/kev-r4/eval-dev/predictions.jsonl
```

**Cổng (ghi vào báo cáo trước khi chạy dev):**

| Cổng | Điều kiện |
|---|---|
| G1 | EM dev Kev ≥ EM dev r4 − 1 điểm % |
| G2 | Tỷ lệ output tọa độ < 0,5% |
| G3 | Tỷ lệ cắt cụt ≤ r4 |

Đạt cả ba thì mở test 116 tài liệu và sang step 6. Không đạt thì dừng lại cùng bạn quyết định: chạy phương án dự phòng, hay giữ format cũ.

---

## Step 6: Confidence (head trên trace → calibration → ngưỡng review)

**Chuyển và sửa**

| Từ (cũ) | Sang (mới) | Sửa gì |
|---|---|---|
| `token_confidence.*` | `confidence/features.py` | Phần phụ thuộc format (mask, kiểu value) lấy từ KevFormat; kích thước summary là tham số |
| `TokenConfidenceHead`, `make_head` | `confidence/heads.py` | Nhận thêm `h_key`/`h_decide`; mode mới `query` (so `h_key` với `h_value`, kiểu query–key của Kev). Bỏ `FeatureHead` |
| `fit_calibration`, `select_calibration` | `confidence/calibration.py` | Giữ nguyên thuật toán |
| `review.*` (policy, ngưỡng, hàng đợi) | `confidence/policy.py`, `eval/review_metrics.py` | **Sửa:** báo thêm `head_error_recall` (không tính các field bắt buộc review) |
| vòng decode trong `cache_token_review.py` | `pipelines/trace_cache.py` | Lưu cả 3 tín hiệu; fingerprint; 1 lần vision forward |
| `token_review_experiment.train_experiment` | `pipelines/head_selection.py`, `pipelines/finalize.py` | **Sửa:** tách thành hai bước, bỏ vòng chờ file 2 giờ; grid và seed lấy từ yaml; bỏ candidate "head có sẵn" (head ngẫu nhiên) |
| `audit_experiment` | `pipelines/audit.py` | Giữ nguyên quy tắc: chỉ mở audit sau khi đã đóng băng |
| (step 3) | `pipelines/infer.py` | Response có `confidence` và `review` thật |

**Kết quả**
- CLI: `cache-traces`, `select-heads`, `finalize`, `audit`.
- Test: `tests/test_confidence.py`, `tests/test_policy.py`, `tests/test_pipelines.py` (cache giả, chạy end-to-end trên CPU).
- Báo cáo `docs/results/kev_confidence.md`.

**Bạn kiểm tra**
```bash
# 1. Phần calibration/policy chuyển đúng: chạy trên cache CŨ của r4
uv run l2r4kie finalize --cache /home/jovyan/bachdx2/l2r4kie/artifacts/token-review-v2/cache --dry-run
# ngưỡng của head attention phải là 0,9597 như CONFIDENCE_REPORT.md
# 2. Pipeline trên extractor Kev
uv run l2r4kie cache-traces --config configs/confidence/kev.yaml --split train   # rồi dev, calibration, risk_validation
uv run l2r4kie select-heads --config configs/confidence/kev.yaml
uv run l2r4kie finalize --config configs/confidence/kev.yaml
uv run l2r4kie cache-traces --config configs/confidence/kev.yaml --split audit
uv run l2r4kie audit --config configs/confidence/kev.yaml
```

**Cổng G4:** review ở mức bắt ≥95% lỗi ≤ 29,9% (r4 + attention). Báo kèm ablation `h_value` / `+h_key` / `+h_decide` / `query`, cận dưới bootstrap, và giới hạn review lý thuyết của từng extractor.

**Trước khi chạy:** chốt cách audit, vì 13 loại hiếm đã hết tài liệu mới. Đề xuất: dùng lại 347 tài liệu audit cũ, ghi rõ là "đã xem", cộng thêm phần tài liệu còn mới.

---

## Step 7: CLI, configs, tài liệu, dọn dẹp

- Rà lại các lệnh CLI đã thêm ở từng step cho nhất quán.
- Configs đặt đường dẫn data/checkpoint ở một chỗ (`paths:`).
- Scripts còn lại: `run_training_sweep.py`, `report_*.py`, `plot_confidence.py`, `audit_isolation.py`, `probe_markers.py`.
- `README.md`: ý tưởng, format, chạy end-to-end. `docs/results/` (cả báo cáo cũ của r4 làm mốc). `docs/history.md` (G1/G2, đối chiếu Kev, bài học).
- **Xong khi:** `pytest` xanh (không cần GPU, < 2 phút); `grep -r /home/jovyan src/` rỗng; chạy lại được chuỗi lệnh trong README.

---

## Không chuyển

- **Code:** `application.predictions(calibrate/mine)`, `domain.negative`, synthetic negatives/mining/ranking/`confidence_only`, `FeatureHead`, `review_optimization.optimize` (phần G2), candidate "head có sẵn".
- **Scripts:** `audit_visual_confidence`, `cache_review_audit`, `cache_review_features`, `finalize_review_policy`, `optimize_review`, `replay_review_serving`, `plot_review_report`, `train_token_review`, `run_token_review_experiment`, `run_confidence_on_winner`, `rerun_confidence_v2`, `finish_training_optimization`, `probe_training_attention`.
- **Configs:** `pilot`, `refine`, `mixed-smoke`, `full`.

## Rủi ro

| Rủi ro | Xử lý |
|---|---|
| Model tiếp tục sinh tọa độ (prior grounding) | Theo dõi ở step 4; dự phòng `close=im_end` |
| Không còn parity với code cũ cho infer/train (format đã đổi) | Test cấu trúc trên tiny model, smoke test, so kết quả với r4 ở step 5–6 |
| `trainable_token_indices` của PEFT 0.18.1 với embedding dùng chung trọng số | K6 mặc định tắt; kiểm tra ở step 4 |
| BF16 không tất định | So chính xác ở FP32; BF16 chỉ so có tolerance |
| Hết tài liệu audit mới cho loại hiếm | Chốt trước ở step 6, báo riêng phần "đã xem" |
| Bản pin transformers 4.57.6 (API KV cache, `get_rope_index`) | Giữ pin; test decode báo ngay khi API đổi |

## Theo dõi

- [x] 0 Nền móng
- [ ] 1 Dữ liệu
- [ ] 2 Tiền xử lý (KevFormat)
- [ ] 3 Infer
- [ ] 4 Train (smoke → chạy thật)
- [ ] 5 Eval (cổng G1–G3)
- [ ] 6 Confidence (cổng G4)
- [ ] 7 CLI, tài liệu, dọn dẹp
