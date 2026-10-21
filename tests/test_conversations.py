"""Chat-style KIE labels read as documents (:mod:`l2r4kie.data.conversations`)."""

from __future__ import annotations

import json
from pathlib import Path

from l2r4kie.data.conversations import conversation_fields, field_descriptions, iter_dataset

PROMPT = '''Trích xuất thông tin từ ảnh tài liệu và trả về JSON theo danh sách trường dưới đây.

Trường cần trích xuất (thụt lề là trường con):
- "Mã mẫu": Mã số biểu mẫu, lấy toàn bộ chuỗi.
- "Hành chính" (object)
  - "Họ và tên": Họ tên viết in hoa.
  - "Giới tính"
- "Xét nghiệm" (array of objects): Mỗi dòng một xét nghiệm
  - "Kết quả": Giữ nguyên ký hiệu
- "Đã ký" (true/false)
- "Số tiền": Số tiền thanh toán.

Quy tắc:
- Chỉ trả về một object JSON hợp lệ.'''


def label(answer: dict, prompt: str = PROMPT) -> dict:
    return {'conversations': [{'role': 'user', 'content': prompt},
                              {'role': 'assistant', 'content': json.dumps(answer, ensure_ascii=False)}]}


def test_field_list_becomes_a_description_tree() -> None:
    assert field_descriptions(PROMPT) == {
        'Mã mẫu': 'Mã số biểu mẫu, lấy toàn bộ chuỗi.',
        'Hành chính': {'Họ và tên': 'Họ tên viết in hoa.', 'Giới tính': None},
        'Xét nghiệm': [{'__desc': 'Mỗi dòng một xét nghiệm', 'Kết quả': 'Giữ nguyên ký hiệu'}],
        'Đã ký': '(true/false)',
        'Số tiền': 'Số tiền thanh toán.'}
    assert field_descriptions('Extract "a", "b" and return JSON.') is None


def test_vietnamese_array_kinds_keep_their_children() -> None:
    prompt = ('- "Thuốc" (mảng object): Mỗi dòng một thuốc\n  - "Tên": Tên thuốc\n'
              '- "Ghi chú" (mảng chuỗi): Từng dòng ghi chú\n- "Có dấu" (true/false): Có dấu đỏ')
    assert field_descriptions(prompt) == {'Thuốc': [{'__desc': 'Mỗi dòng một thuốc', 'Tên': 'Tên thuốc'}],
                                          'Ghi chú': ['Từng dòng ghi chú'], 'Có dấu': '(true/false) Có dấu đỏ'}


def test_answer_leaves_become_fields_with_blank_nulls() -> None:
    answer = {'Mã mẫu': '29/BV-02', 'Hành chính': {'Họ và tên': 'NGUYỄN VĂN A', 'Giới tính': None},
              'Xét nghiệm': [{'Kết quả': '5.2 H'}], 'Đã ký': True, 'Số tiền': 120000}
    fields = {f.id: f for f in conversation_fields(label(answer))}
    assert list(fields) == ['/Mã mẫu', '/Hành chính/Họ và tên', '/Hành chính/Giới tính', '/Xét nghiệm', '/Đã ký',
                            '/Số tiền']
    assert fields['/Hành chính/Họ và tên'].description == 'Họ tên viết in hoa.'
    assert fields['/Hành chính/Giới tính'].description == '/Hành chính/Giới tính'  # no description: the id
    assert fields['/Hành chính/Giới tính'].value == ''
    assert fields['/Xét nghiệm'].kind == 'array' and fields['/Xét nghiệm'].description == 'Mỗi dòng một xét nghiệm'
    assert fields['/Đã ký'].value is True and fields['/Số tiền'].value == '120000'


def test_dataset_keeps_one_variant_and_resolves_moved_paths(tmp_path: Path) -> None:
    (tmp_path / 'labels').mkdir()
    (tmp_path / 'images').mkdir()
    (tmp_path / 'images' / 'p.png').write_bytes(b'x')
    entries = []
    for variant in ('v1', 'v2'):
        path = tmp_path / 'labels' / f'doc__{variant}.json'
        path.write_text(json.dumps(label({'Mã mẫu': variant})), encoding='utf-8')
        entries.append({'label_path': f'/elsewhere/labels/doc__{variant}.json', 'image_paths': ['/elsewhere/p.png']})
    (tmp_path / 'image_label_details.json').write_text(json.dumps(entries), encoding='utf-8')
    documents = list(iter_dataset(tmp_path))
    assert [d.id for d in documents] == [f'{tmp_path.name}__doc__v1']
    assert documents[0].pages == (str(tmp_path / 'images' / 'p.png'),)
    assert documents[0].fields[0].value == 'v1' and documents[0].form == tmp_path.name
