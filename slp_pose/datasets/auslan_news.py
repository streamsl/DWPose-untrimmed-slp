"""Auslan-Daily TV news (News_V1 + News_V2): readable video ids, V1 splits and annotation names.

Under <data root>/Auslan-Daily/ the videos are 'ABC News With Auslan - D M YYYY.mp4' in News_V1/
(2022-2023, sentence annotations in News_V1_Annotation.xlsx, whose Video_Name is 'D_M_YYYY';
'EA<n>' rows belong to the Web News sub-dataset and are not used) and News_V2/ (2023-2025, the
BANZ-FS extension; News_V2_Aligned_VTTS/<file stem>.vtt holds its sign-aligned subtitles). The
video id is 'abc_news_YYYY-MM-DD' (unique: the two subsets share no date). The interpreter stands in
a panel on the right, beside news footage or a studio newsreader who may be larger: the workers pose
with 'right_largest' and the parent derives the persistent right-hand signer track (spec D14).
Standard library only (the workbook is read with zipfile + ElementTree).
"""
from __future__ import annotations

import re
import zipfile
from pathlib import Path
from typing import Dict, List, Tuple
from xml.etree import ElementTree

from . import register
from .base import Dataset, Video

V1_ANNOTATION = 'News_V1_Annotation.xlsx'   # inside Auslan-Daily/
_NAME = re.compile(r'ABC News With Auslan - (\d{1,2}) (\d{1,2}) (\d{4})')
_XLSX = '{http://schemas.openxmlformats.org/spreadsheetml/2006/main}'


@register
class AuslanNews(Dataset):
    """News_V1 + News_V2; split = the V1 workbook's split (None for V2); info: subset, annotation_name."""

    name = 'auslan_news'
    primary_rule = 'signer_track_right'
    extraction_rule = 'right_largest'
    split_order = ('dev', 'test', 'train')

    @property
    def source(self) -> Path:
        return self.data_root / 'Auslan-Daily'

    def out_root(self) -> Path:
        return self.source / 'dwpose'

    def videos(self) -> List[Video]:
        splits = v1_splits(self.source / V1_ANNOTATION)
        out = []
        for path in sorted(self.source.glob('News_V[12]/*.mp4')):
            subset, name = path.parent.name, annotation_name(path)
            out.append(Video(video_id(path), path, splits.get(name) if subset == 'News_V1' else None,
                             dict(subset=subset, annotation_name=name)))
        return out


def _date(path: Path) -> Tuple[int, int, int]:
    match = _NAME.fullmatch(Path(path).stem)
    if match is None:
        raise ValueError(f"{path}: expected 'ABC News With Auslan - D M YYYY.mp4'")
    day, month, year = (int(g) for g in match.groups())
    return day, month, year


def video_id(path: Path) -> str:
    """'ABC News With Auslan - 16 10 2022.mp4' -> 'abc_news_2022-10-16'."""
    day, month, year = _date(path)
    return f'abc_news_{year:04d}-{month:02d}-{day:02d}'


def annotation_name(path: Path) -> str:
    """The video's name in its annotations: 'D_M_YYYY' (the Excel Video_Name) for News_V1, the
    file stem (the VTT name) for News_V2."""
    if Path(path).parent.name == 'News_V1':
        day, month, year = _date(path)
        return f'{day}_{month}_{year}'
    return Path(path).stem


def v1_splits(xlsx: Path) -> Dict[str, str]:
    """{Video_Name: split} of the TV-news rows of the News_V1 annotation workbook (first sheet;
    'EA<n>' Web News rows are skipped). Raises ValueError when a video has rows in more than one
    split."""
    with zipfile.ZipFile(xlsx) as zf:
        shared = [''.join(t.text or '' for t in si.iter(f'{_XLSX}t'))
                  for si in ElementTree.fromstring(zf.read('xl/sharedStrings.xml')).iter(f'{_XLSX}si')]
        sheet = ElementTree.fromstring(zf.read('xl/worksheets/sheet1.xml'))
    rows = [_cells(row, shared) for row in sheet.iter(f'{_XLSX}row')]
    columns = {name: col for col, name in rows[0].items()}
    out: Dict[str, str] = {}
    for row in rows[1:]:
        name, split = row.get(columns['Video_Name'], ''), row.get(columns['split'], '')
        if not name or name.startswith('EA'):
            continue
        if out.setdefault(name, split) != split:
            raise ValueError(f'{xlsx}: video {name!r} has rows in splits {out[name]!r} and {split!r}')
    return out


def _cells(row: ElementTree.Element, shared: List[str]) -> Dict[str, str]:
    """{column letters: text} of one sheet row (shared, inline and plain values)."""
    out = {}
    for cell in row.iter(f'{_XLSX}c'):
        column = re.match(r'[A-Z]+', cell.get('r', '')).group()
        value = cell.find(f'{_XLSX}v')
        if cell.get('t') == 's' and value is not None:
            out[column] = shared[int(value.text)]
        elif value is not None:
            out[column] = value.text or ''
        else:
            out[column] = ''.join(t.text or '' for t in cell.iter(f'{_XLSX}t'))
    return out
