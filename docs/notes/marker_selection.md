# Chọn marker cho KevFormat (09/10/2026)

## Kev làm gì

Kev ([kev/model.py](https://github.com/jaredpalmer/kev/blob/main/kev/model.py)) không thêm token. Nó dùng lại 5 special token ít dùng của Qwen2.5:

```python
SPECIAL = ["<|fim_prefix|>", "<|fim_middle|>", "<|box_start|>", "<|box_end|>", "<|fim_suffix|>"]
#           state             question          option mở        option đóng     decide
```

- Embedding của các token này chỉ được train thêm khi bật tùy chọn `special_embeddings`, bằng `trainable_token_indices` của PEFT.
- Text của người gọi được escape mẫu `<|name|>` trước khi tokenize, nên input không tạo ra được token điều khiển.

## Qwen2-VL-2B có gì

Qwen2-VL-2B không có `<|fim_*|>`, chỉ có 14 special token (id 151643–151656). Ma trận embedding có 151.936 hàng; các hàng dự trữ chưa train có norm 0,435, token thường có norm trung vị 0,600. Embedding và LM head dùng chung trọng số.

| Token | Norm | Cos với hàng chưa train | Đánh giá |
|---|---:|---:|---|
| `object_ref_start` / `object_ref_end` | 0,574 / 0,551 | 0,45 / 0,35 | Đã train rõ ràng. **Dùng cho key.** |
| `box_start` / `box_end` | 0,483 / 0,521 | 0,63 / 0,72 | Đã train. **Dùng cho value**, giống cặp option của Kev. |
| `quad_start` / `quad_end` | 0,442 / 0,501 | 0,77 / 0,63 | Train yếu. Chỉ dự phòng. |
| `vision_start/end`, `vision_pad`, `image_pad`, `video_pad` | 0,435 | 1,00 | Trùng hàng chưa train, và `get_rope_index` dùng các token này để định vị ảnh. **Không được dùng.** |
| `im_start`, `im_end`, `endoftext` | | | Thuộc chat template và padding. Không dùng làm marker. |

## Probe zero-shot

Base model, 24 field của 6 tài liệu test, greedy 40 token ([scripts/probe_markers.py](../../scripts/probe_markers.py)):

| Cặp value | Đóng đúng marker | Sinh tọa độ thay vì value | Special token lạc |
|---|---|---|---|
| `object_ref` + `box` | **24/24** | 24/24 | 0 |
| `object_ref` + `quad` | 0/24 (đóng bằng `box_end`) | 24/24 | 24/24 |

Ví dụ: field "Mẫu số" (GT `01/KBCB`) → `(684,10),(996,101)<|box_end|>`.

## Kết luận

- `box_end` là marker đóng mà base model đã sinh rất ổn định, nên ít lỗi không đóng/cắt cụt. Base model không biết đóng bằng `quad_end`.
- Cái giá: Qwen2-VL được pretrain grounding theo mẫu `<|object_ref_start|>tên<|object_ref_end|><|box_start|>(x1,y1),(x2,y2)<|box_end|>`, nên ban đầu model sinh tọa độ. Fine-tune phải đổi nội dung từ tọa độ sang text. Cần theo dõi tỷ lệ output dạng tọa độ khi train.
- Chưa train mà model đã khoanh được vùng gần đúng cho một số field. Prior này có thể dùng sau cho grounded extraction (sinh box rồi mới sinh value).
- Nếu đổi sang Qwen2.5-VL, có thể dùng nguyên bộ token `fim` như Kev.
- Không cần marker cho tài liệu, vì ảnh đã được bọc bằng `vision_start/end`. Không cần `</end>`, vì `box_end` vừa để dừng vừa làm tín hiệu.
- Đây chỉ là probe nhỏ (24 field), đủ để thấy xu hướng, không phải con số chính xác.
