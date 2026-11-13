from __future__ import annotations

import math
import struct
import zipfile
from io import BytesIO
from pathlib import Path

import numpy as np
import pandas as pd

ID_COL = 'utility_cohort_id'
TIME_COL = 'decision_date'
TARGET = 'deteriorated'
CARRY = ['utility_id', 'forecast_year']
FEATURES = [
    'saidi_nomed_lag1',
    'saidi_nomed_lag2',
    'saidi_nomed_lag3',
    'saidi_nomed_trend',
    'saidi_with_med_lag1',
    'saifi_nomed_lag1',
    'saifi_with_med_lag1',
    'caidi_nomed_lag1',
    'customers_lag1',
    'reporting_standard',
    'ownership_code',
    'state',
]
COLUMNS = [ID_COL, TIME_COL] + CARRY + FEATURES + [TARGET]
SPLIT_EVAL_START = '2020-01-01T00:00:00.000'
SPLIT_HOLDOUT_START = '2022-01-01T00:00:00.000'
FEATURE_OFFSETS = {
    'saidi_nomed_lag1': -1,
    'saidi_nomed_lag2': -2,
    'saidi_nomed_lag3': -3,
    'saidi_nomed_trend': -1,
    'saidi_with_med_lag1': -1,
    'saifi_nomed_lag1': -1,
    'saifi_with_med_lag1': -1,
    'caidi_nomed_lag1': -1,
    'customers_lag1': -1,
    'reporting_standard': -1,
    'ownership_code': -1,
    'state': -1,
}
ALLOWED_OWNERSHIP = {
    'Investor Owned',
    'Municipal',
    'Cooperative',
    'Political Subdivision',
    'Federal',
    'State',
}
OWNERSHIP_CODES = {
    '1': 'Investor Owned',
    '2': 'Municipal',
    '3': 'Cooperative',
    '4': 'Federal',
    '5': 'Political Subdivision',
    '6': 'State',
}
URLS = {
    2013: 'https://www.eia.gov/electricity/data/eia861/archive/zip/f8612013.zip',
    2014: 'https://www.eia.gov/electricity/data/eia861/archive/zip/f8612014.zip',
    2015: 'https://www.eia.gov/electricity/data/eia861/archive/zip/f8612015.zip',
    2016: 'https://www.eia.gov/electricity/data/eia861/archive/zip/f8612016.zip',
    2017: 'https://www.eia.gov/electricity/data/eia861/archive/zip/f8612017.zip',
    2018: 'https://www.eia.gov/electricity/data/eia861/archive/zip/f8612018.zip',
    2019: 'https://www.eia.gov/electricity/data/eia861/archive/zip/f8612019.zip',
    2020: 'https://www.eia.gov/electricity/data/eia861/archive/zip/f8612020.zip',
    2021: 'https://www.eia.gov/electricity/data/eia861/archive/zip/f8612021.zip',
    2022: 'https://www.eia.gov/electricity/data/eia861/archive/zip/f8612022.zip',
    2023: 'https://www.eia.gov/electricity/data/eia861/archive/zip/f8612023.zip',
    2024: 'https://www.eia.gov/electricity/data/eia861/archive/zip/f8612024.zip',
    2025: 'https://www.eia.gov/electricity/data/eia861/zip/f8612025.zip',
}
_OLE_SIG = bytes([0xD0, 0xCF, 0x11, 0xE0, 0xA1, 0xB1, 0x1A, 0xE1])


def _alt_urls(year):
    primary = URLS[year]
    if '/archive/zip/' in primary:
        return [primary.replace('/archive/zip/', '/zip/')]
    return [primary.replace('/zip/', '/archive/zip/')]


def _text(v):
    if v is None:
        return ''
    if isinstance(v, float) and math.isnan(v):
        return ''
    if isinstance(v, bool):
        return 'yes' if v else 'no'
    try:
        if pd.isna(v):
            return ''
    except Exception:
        pass
    return str(v).strip()


def _norm(v):
    s = _text(v).lower()
    chars = [ch if ch.isalnum() else ' ' for ch in s]
    return ' '.join(''.join(chars).split())


def _num(v):
    if v is None or isinstance(v, bool):
        return np.nan
    if isinstance(v, (int, float, np.integer, np.floating)):
        x = float(v)
        return x if math.isfinite(x) else np.nan
    s = str(v).strip().replace(',', '')
    if not s:
        return np.nan
    neg = False
    if s.startswith('(') and s.endswith(')'):
        neg = True
        s = s[1:-1].strip()
    s = s.rstrip('*').strip()
    if s in ('', '.', '-', '--', 'NA', 'N/A', 'n/a', 'NM', 'W', 'Q'):
        return np.nan
    try:
        x = float(s)
    except Exception:
        return np.nan
    if not math.isfinite(x):
        return np.nan
    return -x if neg else x


def _clean_id(v):
    s = _text(v)
    if not s:
        return None
    if s.endswith('.0') and s[:-2].isdigit():
        s = s[:-2]
    if s.isdigit():
        return s
    return None


def _state(v):
    s = _text(v).strip()
    if len(s) == 2 and s.isalpha():
        return s.upper()
    return None


def _ownership(v):
    s = _text(v).strip()
    if s in OWNERSHIP_CODES:
        return OWNERSHIP_CODES[s]
    n = _norm(v)
    if not n:
        return None
    if 'investor' in n or n == 'iou' or n == 'investor owned':
        return 'Investor Owned'
    if 'municipal' in n or n == 'muni':
        return 'Municipal'
    if 'cooperative' in n or 'co op' in n or n == 'coop':
        return 'Cooperative'
    if 'political' in n:
        return 'Political Subdivision'
    if 'federal' in n:
        return 'Federal'
    if n == 'state' or n.startswith('state '):
        return 'State'
    return None


def _std(v):
    n = _norm(v)
    if not n:
        return None
    if n in ('yes', 'y', 'true', '1'):
        return 'IEEE'
    if n in ('no', 'n', 'false', '0'):
        return 'OTHER'
    if 'ieee' in n and 'non' not in n:
        return 'IEEE'
    if 'other' in n or 'non' in n or 'internal' in n:
        return 'OTHER'
    return None


def _is_uid_label(n):
    if 'name' in n:
        return False
    return ('utility' in n and ('number' in n or ' id' in n or n.endswith(' id'))) or n == 'eia id'


def _is_customer_label(n):
    if 'percent' in n or 'sales' in n or 'revenue' in n:
        return False
    return 'customer' in n or 'meter' in n


def _variant(n):
    if 'without' in n or 'excl' in n or 'no med' in n or 'no major' in n:
        return 'without'
    if 'with' in n or 'major event' in n or ' med' in n:
        return 'with'
    return None


def _block(n):
    if 'non ieee' in n:
        return 'OTHER'
    if 'ieee' in n:
        return 'IEEE'
    if 'other' in n:
        return 'OTHER'
    return 'COMMON'


def _get(row, idx):
    if idx is None or idx < 0 or idx >= len(row):
        return None
    return row[idx]


def _rk(v):
    if v & 0x02:
        i = v >> 2
        if i & 0x20000000:
            i -= 0x40000000
        x = float(i)
    else:
        bits = (v & 0xFFFFFFFC) << 32
        x = struct.unpack('<d', struct.pack('<Q', bits & 0xFFFFFFFFFFFFFFFF))[0]
    if v & 0x01:
        x = x / 100.0
    return x


def _ole_stream(data):
    if len(data) < 512 or data[:8] != _OLE_SIG:
        return None
    sec = 1 << struct.unpack_from('<H', data, 0x1E)[0]
    mini = 1 << struct.unpack_from('<H', data, 0x20)[0]
    n_fat = struct.unpack_from('<I', data, 0x2C)[0]
    dir_start = struct.unpack_from('<I', data, 0x30)[0]
    mini_cutoff = struct.unpack_from('<I', data, 0x38)[0]
    mini_fat_start = struct.unpack_from('<I', data, 0x3C)[0]
    difat_start = struct.unpack_from('<I', data, 0x44)[0]
    n_difat = struct.unpack_from('<I', data, 0x48)[0]
    difat = list(struct.unpack_from('<109I', data, 0x4C))
    sid = difat_start
    guard = 0
    while sid not in (0xFFFFFFFE, 0xFFFFFFFF) and guard < n_difat + 8:
        off = (sid + 1) * sec
        chunk = data[off:off + sec]
        if len(chunk) < sec:
            break
        vals = list(struct.unpack_from('<' + str(sec // 4) + 'I', chunk, 0))
        difat.extend(vals[:-1])
        sid = vals[-1]
        guard += 1
    fat = []
    for fsid in difat[:max(0, int(n_fat))]:
        if fsid in (0xFFFFFFFE, 0xFFFFFFFF):
            continue
        off = (fsid + 1) * sec
        chunk = data[off:off + sec]
        if len(chunk) < sec:
            continue
        fat.extend(struct.unpack_from('<' + str(sec // 4) + 'I', chunk, 0))
    if not fat:
        return None

    def chain(start):
        out = []
        s = start
        seen = set()
        while s not in (0xFFFFFFFE, 0xFFFFFFFF) and s < len(fat) and s not in seen and len(out) < 1000000:
            seen.add(s)
            out.append(s)
            s = fat[s]
        return out

    def read_stream(start, size):
        buf = bytearray()
        for s in chain(start):
            off = (s + 1) * sec
            buf += data[off:off + sec]
            if size is not None and len(buf) >= size:
                break
        if size is None:
            return bytes(buf)
        return bytes(buf[:size])

    dirdata = read_stream(dir_start, None)
    entries = []
    for i in range(0, len(dirdata) - 127, 128):
        e = dirdata[i:i + 128]
        nlen = struct.unpack_from('<H', e, 0x40)[0]
        typ = e[0x42]
        if typ == 0:
            continue
        nm = e[:max(0, nlen - 2)].decode('utf-16-le', 'ignore')
        start = struct.unpack_from('<I', e, 0x74)[0]
        size = struct.unpack_from('<Q', e, 0x78)[0]
        entries.append((nm, typ, start, size))
    root = None
    target = None
    for nm, typ, start, size in entries:
        if typ == 5 and root is None:
            root = (start, size)
        if typ == 2 and nm in ('Workbook', 'Book'):
            target = (start, size)
    if target is None or root is None:
        return None
    start, size = target
    if size >= mini_cutoff:
        return read_stream(start, size)
    ms = read_stream(root[0], root[1])
    minifat = []
    if mini_fat_start not in (0xFFFFFFFE, 0xFFFFFFFF):
        raw = read_stream(mini_fat_start, None)
        if raw:
            minifat = list(struct.unpack_from('<' + str(len(raw) // 4) + 'I', raw, 0))
    buf = bytearray()
    s = start
    guard2 = 0
    while s not in (0xFFFFFFFE, 0xFFFFFFFF) and s < len(minifat) and guard2 < 1000000:
        off = s * mini
        buf += ms[off:off + mini]
        s = minifat[s]
        guard2 += 1
        if len(buf) >= size:
            break
    return bytes(buf[:size])


def _parse_sst(chunks):
    out = []
    chunks = [c for c in chunks if c]
    if not chunks:
        return out
    n = len(chunks)
    ci = 0
    pos = 8
    guard = 0

    def ensure():
        nonlocal ci, pos
        while ci < n and pos >= len(chunks[ci]):
            ci += 1
            pos = 0
        return ci < n

    while guard < 500000:
        guard += 1
        if not ensure():
            break
        ch = chunks[ci]
        if pos + 3 > len(ch):
            ci += 1
            pos = 0
            continue
        cch = ch[pos] | (ch[pos + 1] << 8)
        pos += 2
        if not ensure():
            break
        flags = chunks[ci][pos]
        pos += 1
        high = flags & 0x01
        nrich = 0
        cbext = 0
        if flags & 0x08:
            if not ensure():
                break
            c2 = chunks[ci]
            if pos + 2 > len(c2):
                break
            nrich = c2[pos] | (c2[pos + 1] << 8)
            pos += 2
        if flags & 0x04:
            if not ensure():
                break
            c2 = chunks[ci]
            if pos + 4 > len(c2):
                break
            cbext = struct.unpack_from('<I', c2, pos)[0]
            pos += 4
        parts = []
        remaining = cch
        while remaining > 0:
            if not ensure():
                break
            ch = chunks[ci]
            avail = len(ch) - pos
            if avail <= 0:
                ci += 1
                pos = 0
                if ci >= n:
                    break
                high = chunks[ci][pos] & 0x01
                pos += 1
                continue
            take = min(remaining, avail // 2) if high else min(remaining, avail)
            if take <= 0:
                ci += 1
                pos = 0
                if ci >= n:
                    break
                high = chunks[ci][pos] & 0x01
                pos += 1
                continue
            if high:
                seg = ch[pos:pos + 2 * take]
                parts.append(seg.decode('utf-16-le', 'ignore'))
                pos += 2 * take
            else:
                seg = ch[pos:pos + take]
                parts.append(seg.decode('latin-1', 'ignore'))
                pos += take
            remaining -= take
        out.append(''.join(parts))
        skip = 4 * nrich + cbext
        while skip > 0:
            if not ensure():
                break
            ch = chunks[ci]
            can = min(skip, len(ch) - pos)
            if can <= 0:
                ci += 1
                pos = 0
                break
            pos += can
            skip -= can
    return out


def _cell(cells, typ, data, sst):
    if typ == 0x00FD and len(data) >= 10:
        r, c, _xf, idx = struct.unpack('<HHHI', data[:10])
        cells[(r, c)] = sst[idx] if idx < len(sst) else None
    elif typ == 0x0204 and len(data) >= 9:
        r, c, _xf, cch = struct.unpack('<HHHH', data[:8])
        flags = data[8]
        if flags & 0x01:
            raw = data[9:9 + 2 * cch]
            cells[(r, c)] = raw.decode('utf-16-le', 'ignore')
        else:
            raw = data[9:9 + cch]
            cells[(r, c)] = raw.decode('latin-1', 'ignore')
    elif typ == 0x0203 and len(data) >= 14:
        r, c, _xf = struct.unpack('<HHH', data[:6])
        cells[(r, c)] = struct.unpack('<d', data[6:14])[0]
    elif typ == 0x027E and len(data) >= 10:
        r, c, _xf, rk = struct.unpack('<HHHI', data[:10])
        cells[(r, c)] = _rk(rk)
    elif typ == 0x00BD and len(data) >= 6:
        r, c0 = struct.unpack('<HH', data[:4])
        ncell = (len(data) - 6) // 6
        for i in range(ncell):
            off = 4 + i * 6
            rk = struct.unpack('<I', data[off + 2:off + 6])[0]
            cells[(r, c0 + i)] = _rk(rk)
    elif typ == 0x0006 and len(data) >= 14:
        r, c, _xf = struct.unpack('<HHH', data[:6])
        if data[12] == 0xFF and data[13] == 0xFF:
            if data[6] == 1:
                cells[(r, c)] = bool(data[8])
            else:
                cells[(r, c)] = None
        else:
            cells[(r, c)] = struct.unpack('<d', data[6:14])[0]
    elif typ == 0x0205 and len(data) >= 8:
        r, c, _xf = struct.unpack('<HHH', data[:6])
        cells[(r, c)] = None


def _matrix(cells):
    if not cells:
        return []
    maxr = max(r for r, _c in cells)
    maxc = max(c for _r, c in cells)
    maxr = min(maxr, 200000)
    maxc = min(maxc, 300)
    mat = [[None] * (maxc + 1) for _ in range(maxr + 1)]
    for (r, c), v in cells.items():
        if r <= maxr and c <= maxc:
            mat[r][c] = v
    return mat


def _biff_workbook(wb):
    sst = []
    sheets = {}
    names = []
    cur = None
    sheet_count = 0
    pos = 0
    L = len(wb)
    while pos + 4 <= L:
        typ = wb[pos] | (wb[pos + 1] << 8)
        ln = wb[pos + 2] | (wb[pos + 3] << 8)
        data = wb[pos + 4:pos + 4 + ln]
        nxt = pos + 4 + ln
        if typ == 0x00FC:
            chunks = [data]
            p = nxt
            while p + 4 <= L:
                t2 = wb[p] | (wb[p + 1] << 8)
                l2 = wb[p + 2] | (wb[p + 3] << 8)
                if t2 != 0x003C:
                    break
                chunks.append(wb[p + 4:p + 4 + l2])
                p += 4 + l2
            pos = p
            try:
                sst = _parse_sst(chunks)
            except Exception:
                sst = []
            continue
        pos = nxt
        if typ == 0x0085 and len(data) >= 8:
            cch = data[6]
            flags = data[7]
            if flags & 0x01:
                names.append(data[8:8 + 2 * cch].decode('utf-16-le', 'ignore'))
            else:
                names.append(data[8:8 + cch].decode('latin-1', 'ignore'))
        elif typ == 0x0809 and len(data) >= 4:
            dt = struct.unpack_from('<H', data, 2)[0]
            if dt == 0x0010:
                nm = names[sheet_count] if sheet_count < len(names) else 'sheet' + str(sheet_count)
                sheet_count += 1
                cur = {}
                sheets[nm] = cur
            else:
                cur = None
        elif cur is not None:
            try:
                _cell(cur, typ, data, sst)
            except Exception:
                pass
    return {k: _matrix(v) for k, v in sheets.items()}


def _read_xlsx_sheets(data):
    from openpyxl import load_workbook
    wb = load_workbook(BytesIO(data), data_only=True, read_only=False)
    out = {}
    for ws in wb.worksheets:
        nr = ws.max_row or 0
        nc = ws.max_column or 0
        nr = min(nr, 200000)
        nc = min(nc, 200)
        if nr < 1 or nc < 1:
            out[ws.title] = []
            continue
        mat = [[None] * nc for _ in range(nr)]
        i = 0
        for row in ws.iter_rows(min_row=1, max_row=nr, min_col=1, max_col=nc, values_only=True):
            for j, v in enumerate(row):
                if j < nc:
                    mat[i][j] = v
            i += 1
            if i >= nr:
                break
        try:
            for rng in ws.merged_cells.ranges:
                r0 = rng.min_row - 1
                r1 = min(rng.max_row - 1, nr - 1)
                c0 = rng.min_col - 1
                c1 = min(rng.max_col - 1, nc - 1)
                if r0 < 0 or c0 < 0 or r0 >= nr or c0 >= nc:
                    continue
                v = mat[r0][c0]
                for rr in range(r0, r1 + 1):
                    for cc in range(c0, c1 + 1):
                        mat[rr][cc] = v
        except Exception:
            pass
        out[ws.title] = mat
    wb.close()
    return out


def _read_html_sheets(data):
    from bs4 import BeautifulSoup
    soup = BeautifulSoup(data, 'lxml')
    out = {}
    for ti, table in enumerate(soup.find_all('table')):
        mat = []
        for tr in table.find_all('tr'):
            cells = tr.find_all(['td', 'th'])
            mat.append([c.get_text(' ', strip=True) for c in cells])
        out['table' + str(ti)] = mat
    return out


def _read_workbook_bytes(data, member):
    low = member.lower()
    if data[:2] == b'PK' or low.endswith('.xlsx') or low.endswith('.xlsm'):
        return _read_xlsx_sheets(data)
    if data[:8] == _OLE_SIG:
        wb = _ole_stream(data)
        if wb:
            return _biff_workbook(wb)
    low2 = data[:400000].lower()
    if b'<html' in low2 or b'<table' in low2:
        return _read_html_sheets(data)
    raise AssertionError('Workbook bytes for ' + str(member) + ' were not readable as xlsx, xls or html. First bytes ' + repr(data[:8]))


def _labels(mat, h):
    ncols = 0
    for row in mat:
        if len(row) > ncols:
            ncols = len(row)
    labels = []
    for c in range(ncols):
        parts = []
        for r in range(max(0, h - 3), h + 1):
            t = _text(_get(mat[r], c))
            if not t:
                continue
            nt = _norm(t)
            if 'energy information administration' in nt or 'form eia' in nt or nt.startswith('page '):
                continue
            if t not in parts:
                parts.append(t)
        labels.append(' | '.join(parts))
    return labels


def _find_col(labels, pred):
    for i, label in enumerate(labels):
        if pred(_norm(label)):
            return i
    return None


def _fail(msg, labels, member):
    text = msg + ' | selected archive member ' + str(member) + ' | observed headers ' + ' || '.join(str(l) for l in labels)
    raise AssertionError(text[:4000])


def _probe_header(mat):
    limit = min(len(mat), 40)
    for h in range(0, limit):
        labels = _labels(mat, h)
        if not labels:
            continue
        uid_idx = _find_col(labels, _is_uid_label)
        if uid_idx is None:
            continue
        blob = ' '.join(_norm(l) for l in labels)
        if 'saidi' not in blob:
            continue
        valid = 0
        for row in mat[h + 1:h + 30]:
            if _clean_id(_get(row, uid_idx)) is not None:
                valid += 1
        if valid >= 5:
            return h, labels
    return None


def _pick_value(row, cands, std):
    if not cands:
        return np.nan, None
    if std in ('IEEE', 'OTHER'):
        pref = [(i, b) for i, b in cands if b == std]
        if pref:
            for i, b in pref:
                v = _num(_get(row, i))
                if math.isfinite(v):
                    return v, b
            return np.nan, std
        common = [(i, b) for i, b in cands if b == 'COMMON']
        if common:
            for i, b in common:
                v = _num(_get(row, i))
                if math.isfinite(v):
                    return v, 'COMMON'
            return np.nan, 'COMMON'
        return np.nan, 'MISMATCH'
    for i, b in cands:
        v = _num(_get(row, i))
        if math.isfinite(v):
            return v, b
    return np.nan, None


def _parse_reliability(sheets, year, member):
    picked_name = None
    mat = None
    h = None
    labels = None
    tried = []
    for sname, smat in sheets.items():
        if not smat:
            continue
        got = _probe_header(smat)
        if got is None:
            tried.append(sname)
            continue
        picked_name = sname
        mat = smat
        h, labels = got
        break
    if mat is None:
        for sname, smat in sheets.items():
            if smat:
                _fail('No worksheet with a utility identifier header and SAIDI columns for ' + str(year) + '. Sheets tried ' + ', '.join(tried), _labels(smat, min(5, max(0, len(smat) - 1))), member)
        raise AssertionError('Workbook for ' + str(year) + ' had no sheet content in member ' + str(member))
    uid_idx = _find_col(labels, _is_uid_label)
    state_idx = _find_col(labels, lambda n: n == 'state' or n.startswith('state ') or ' state ' in n)
    own_idx = _find_col(labels, lambda n: 'ownership' in n or 'owner type' in n)
    std_idx = _find_col(labels, lambda n: ('standard' in n or 'method' in n or n == 'ieee') and not any(m in n for m in ('saidi', 'saifi', 'caidi')))
    customer_cols = []
    for i, label in enumerate(labels):
        n = _norm(label)
        if _is_customer_label(n):
            customer_cols.append((i, _block(n)))
    metric_cols = {}
    bad = []
    if uid_idx is None:
        bad.append('utility number or utility id')
    if not customer_cols:
        bad.append('customer or meter count')
    for i, label in enumerate(labels):
        n = _norm(label)
        for metric in ('saidi', 'saifi', 'caidi'):
            if metric in n:
                variant = _variant(n)
                if variant is None:
                    continue
                block = _block(n)
                metric_cols.setdefault(metric, {}).setdefault(variant, []).append((i, block))
    for metric, variant in (('saidi', 'without'), ('saidi', 'with'), ('saifi', 'without'), ('saifi', 'with')):
        if not metric_cols.get(metric, {}).get(variant):
            bad.append(metric + ' ' + variant + ' major event days')
    if bad:
        _fail('Missing required Reliability fields for ' + str(year) + ': ' + ', '.join(bad), labels, member)
    all_rows = []
    for row in mat[h + 1:]:
        uid = _clean_id(_get(row, uid_idx))
        if uid is None:
            continue
        block_counts = {}
        for metric in ('saidi', 'saifi', 'caidi'):
            for variant, cols in metric_cols.get(metric, {}).items():
                for idx, block in cols:
                    if math.isfinite(_num(_get(row, idx))):
                        block_counts[block] = block_counts.get(block, 0) + 1
        std_explicit = _std(_get(row, std_idx)) if std_idx is not None else None
        block_set = [b for b in block_counts if b in ('IEEE', 'OTHER')]
        conflict = False
        if std_explicit is not None:
            std = std_explicit
            if block_set and std_explicit not in block_set:
                conflict = True
        elif len(block_set) == 1:
            std = block_set[0]
        elif len(block_set) > 1:
            std = None
            conflict = True
        else:
            std = None
        rec_std = std if std in ('IEEE', 'OTHER') and not conflict else None
        std_pick = rec_std
        cust, _ = _pick_value(row, customer_cols, std_pick)
        state = _state(_get(row, state_idx)) if state_idx is not None else None
        own = _ownership(_get(row, own_idx)) if own_idx is not None else None
        sn, bsn = _pick_value(row, metric_cols.get('saidi', {}).get('without', []), std_pick)
        sw, bsw = _pick_value(row, metric_cols.get('saidi', {}).get('with', []), std_pick)
        fn, bfn = _pick_value(row, metric_cols.get('saifi', {}).get('without', []), std_pick)
        fw, bfw = _pick_value(row, metric_cols.get('saifi', {}).get('with', []), std_pick)
        cn, bcn = _pick_value(row, metric_cols.get('caidi', {}).get('without', []), std_pick)
        if (not math.isfinite(cn)) and math.isfinite(sn) and math.isfinite(fn) and fn > 0:
            cn = sn / fn
        all_rows.append({
            'utility_id': uid,
            'state': state,
            'ownership_code': own,
            'reporting_standard': rec_std,
            'standard_conflict': conflict,
            'customers': cust,
            'saidi_nomed': sn,
            'saidi_with_med': sw,
            'saifi_nomed': fn,
            'saifi_with_med': fw,
            'caidi_nomed': cn,
            'block_saidi_nomed': bsn,
            'block_saidi_with_med': bsw,
            'block_saifi_nomed': bfn,
            'block_saifi_with_med': bfw,
            'block_caidi_nomed': bcn,
        })
    counts = {}
    for rec in all_rows:
        counts[rec['utility_id']] = counts.get(rec['utility_id'], 0) + 1
    records = {}
    for rec in all_rows:
        if counts[rec['utility_id']] == 1:
            records[rec['utility_id']] = rec
    has_other_block = False
    for metric, variants in metric_cols.items():
        for variant, cols in variants.items():
            if any(b == 'OTHER' for _i, b in cols):
                has_other_block = True
    meta = {
        'member': member,
        'sheet': picked_name,
        'headers': labels,
        'n_rows': len(all_rows),
        'n_unique': len(records),
        'ownership_idx': own_idx,
        'state_idx': state_idx,
        'standard_idx': std_idx,
        'customer_cols': len(customer_cols),
        'has_other_block': has_other_block,
    }
    return records, all_rows, meta


def _load_attribute_map(zf, year, attr):
    cands = []
    for n in zf.namelist():
        b = Path(n).name.lower()
        if b.startswith('~$'):
            continue
        if not b.endswith(('.xlsx', '.xlsm', '.xls')):
            continue
        if 'reliab' in b:
            continue
        rank = 0 if 'utility' in b else 1
        cands.append((rank, n))
    cands.sort()
    for _rank, n in cands[:4]:
        try:
            data = zf.read(n)
            sheets = _read_workbook_bytes(data, n)
        except Exception:
            continue
        for _sname, mat in sheets.items():
            if not mat:
                continue
            for h in range(0, min(len(mat), 30)):
                labels = _labels(mat, h)
                uid_idx = _find_col(labels, _is_uid_label)
                if uid_idx is None:
                    continue
                if attr == 'ownership':
                    aidx = _find_col(labels, lambda x: 'ownership' in x or 'owner type' in x)
                else:
                    aidx = _find_col(labels, lambda x: x == 'state' or x.startswith('state ') or ' state ' in x)
                if aidx is None:
                    continue
                out = {}
                conflicts = set()
                for row in mat[h + 1:]:
                    uid = _clean_id(_get(row, uid_idx))
                    if uid is None:
                        continue
                    raw = _get(row, aidx)
                    val = _ownership(raw) if attr == 'ownership' else _state(raw)
                    if val is None:
                        continue
                    if uid in out and out[uid] != val:
                        conflicts.add(uid)
                    else:
                        out[uid] = val
                for u in conflicts:
                    out.pop(u, None)
                if out:
                    return out, n
    return {}, None


def _pick_members(zf):
    def rank(n):
        b = Path(n).name.lower()
        if b.startswith('~$'):
            return 99
        if not b.endswith(('.xlsx', '.xlsm', '.xls')):
            return 99
        if 'reliab' in b:
            return 0
        if 'utility' in b:
            return 1
        return 2
    names = [n for n in zf.namelist() if rank(n) < 99]
    names.sort(key=lambda n: (rank(n), len(n)))
    return names


def _load_year(zf, year):
    cands = _pick_members(zf)
    if not cands:
        raise AssertionError('No Excel member in annual ZIP for ' + str(year) + '. Members ' + ' | '.join(zf.namelist())[:1800])
    errors = []
    for n in cands[:6]:
        try:
            data = zf.read(n)
            sheets = _read_workbook_bytes(data, n)
        except Exception as exc:
            errors.append(str(n) + ': ' + type(exc).__name__ + ' ' + str(exc)[:200])
            continue
        try:
            return _parse_reliability(sheets, year, n)
        except AssertionError as exc:
            errors.append(str(n) + ': ' + str(exc)[:400])
            continue
    raise AssertionError('No Reliability schedule could be parsed for ' + str(year) + '. Attempts ' + ' ; '.join(errors)[:3500])


def _fetch_year(fetch, cache, year):
    urls = [URLS[year]] + _alt_urls(year)
    errors = []
    for i, u in enumerate(urls):
        name = 'f861' + str(year) + chr(97 + i) + '.zip'
        try:
            p = Path(fetch(u, cache, name))
        except Exception as exc:
            errors.append(str(u) + ' raised ' + type(exc).__name__ + ' ' + str(exc)[:180])
            continue
        try:
            with open(p, 'rb') as fh:
                head = fh.read(4)
        except Exception as exc:
            errors.append(str(u) + ' could not be read back: ' + str(exc)[:120])
            continue
        if head[:2] != b'PK':
            errors.append(str(u) + ' returned bytes starting ' + repr(head[:4]))
            continue
        return p, u
    raise AssertionError('No usable annual ZIP for ' + str(year) + '. Attempts ' + ' ; '.join(errors)[:1500])


def _weighted(rows, key, state=None, only_ieee=True):
    num = 0.0
    den = 0.0
    for rec in rows:
        if only_ieee and rec.get('reporting_standard') != 'IEEE':
            continue
        if state is not None and rec.get('state') != state:
            continue
        v = rec.get(key)
        c = rec.get('customers')
        try:
            v = float(v)
            c = float(c)
        except Exception:
            continue
        if math.isfinite(v) and math.isfinite(c) and c > 0:
            num += v * c
            den += c
    if den <= 0:
        return float('nan')
    return num / den


def _write(frame, path):
    out = frame[COLUMNS].copy()
    out[TARGET] = out[TARGET].astype(int)
    out.to_csv(path, index=False)


def build(output: Path, cache: Path):
    from source_io import fetch
    output = Path(output)
    cache = Path(cache)
    (output / 'public').mkdir(parents=True, exist_ok=True)
    (output / 'private').mkdir(parents=True, exist_ok=True)
    cache.mkdir(parents=True, exist_ok=True)
    year_rows = {}
    year_all = {}
    year_meta = {}
    for year in range(2013, 2026):
        zpath, used_url = _fetch_year(fetch, cache, year)
        try:
            zf = zipfile.ZipFile(zpath)
        except Exception as exc:
            raise AssertionError('Annual ZIP for ' + str(year) + ' from ' + str(used_url) + ' could not be opened: ' + str(exc)[:300])
        with zf:
            try:
                _records, all_rows, meta = _load_year(zf, year)
            except AssertionError as exc:
                raise AssertionError('Reliability parse failed for ' + str(year) + ' from ' + str(used_url) + ': ' + str(exc)[:2500])
            if meta['ownership_idx'] is None or any(r['ownership_code'] is None for r in all_rows):
                omap, _om = _load_attribute_map(zf, year, 'ownership')
                for r in all_rows:
                    if r['ownership_code'] is None:
                        r['ownership_code'] = omap.get(r['utility_id'])
            if meta['state_idx'] is None or any(r['state'] is None for r in all_rows):
                smap, _sm = _load_attribute_map(zf, year, 'state')
                for r in all_rows:
                    if r['state'] is None:
                        r['state'] = smap.get(r['utility_id'])
            counts = {}
            for r in all_rows:
                counts[r['utility_id']] = counts.get(r['utility_id'], 0) + 1
            records = {r['utility_id']: r for r in all_rows if counts[r['utility_id']] == 1}
        year_rows[year] = records
        year_all[year] = all_rows
        year_meta[year] = meta
    checks = []
    members = {str(y): year_meta[y]['member'] for y in sorted(year_meta)}
    checks.append({
        'check': 'each annual publisher ZIP yielded a Reliability schedule member',
        'observed': ' ; '.join(k + ' from ' + v for k, v in members.items()),
        'expected': 'one Reliability workbook per annual ZIP for 2013 through 2025',
        'passed': len(members) == 13 and all(bool(v) for v in members.values()),
    })
    row_counts = {str(y): year_meta[y]['n_rows'] for y in sorted(year_meta)}
    checks.append({
        'check': 'utility-year records parsed from the Reliability workbooks',
        'observed': ' ; '.join(k + ' -> ' + str(v) for k, v in row_counts.items()),
        'expected': 'at least one valid utility number row in every vintage',
        'passed': len(row_counts) == 13 and all(v > 0 for v in row_counts.values()),
    })
    field_trace = {}
    for y in sorted(year_meta):
        m = year_meta[y]
        field_trace[str(y)] = 'sheet ' + str(m['sheet']) + ', rows ' + str(m['n_rows']) + ', customer columns ' + str(m['customer_cols']) + ', state column ' + ('found' if m['state_idx'] is not None else 'absent') + ', ownership column ' + ('found' if m['ownership_idx'] is not None else 'absent') + ', standard column ' + ('found' if m['standard_idx'] is not None else 'absent')
    checks.append({
        'check': 'utility identifier, customer count, SAIDI and SAIFI with and without major event day fields located by header pattern in every vintage',
        'observed': ' ; '.join(k + ' -> ' + v for k, v in field_trace.items()),
        'expected': 'every vintage exposes a utility identifier, a customer or meter count, and both with-MED and without-MED SAIDI and SAIFI columns',
        'passed': all(year_meta[y]['n_rows'] > 0 and year_meta[y]['customer_cols'] > 0 for y in year_meta),
    })
    w24 = _weighted(year_all[2024], 'saidi_with_med')
    n24 = _weighted(year_all[2024], 'saidi_nomed')
    if not math.isfinite(w24) or not math.isfinite(n24):
        w24 = _weighted(year_all[2024], 'saidi_with_med', only_ieee=False)
        n24 = _weighted(year_all[2024], 'saidi_nomed', only_ieee=False)
    checks.append({
        'check': 'unit and frame check: 2024 national customer-weighted SAIDI in minutes, with and without major event days',
        'observed': 'with major event days ' + str(round(float(w24), 2)) + ' ; without major event days ' + str(round(float(n24), 2)),
        'expected': 'publisher state and national table reports about 662.6 minutes with major event days and 131.6 minutes without for 2024; reconstruction accepted inside a broad minutes-per-customer band with the with-MED aggregate above the without-MED aggregate',
        'passed': math.isfinite(w24) and math.isfinite(n24) and 50.0 <= n24 <= 1200.0 and 100.0 <= w24 <= 3000.0 and w24 >= n24,
        'kind': 'unit',
    })
    fl24 = _weighted(year_all[2024], 'saidi_with_med', 'FL')
    fl25 = _weighted(year_all[2025], 'saidi_with_med', 'FL')
    checks.append({
        'check': 'unit and frame check: Florida SAIDI with major event days, 2024 against 2025',
        'observed': '2024 ' + str(round(float(fl24), 2)) + ' ; 2025 ' + str(round(float(fl25), 2)),
        'expected': 'publisher state evidence reports about 1487.6 minutes for Florida in 2024 against about 72.7 minutes in 2025, so the 2024 aggregate should exceed the 2025 aggregate',
        'passed': math.isfinite(fl24) and math.isfinite(fl25) and fl24 > fl25,
        'kind': 'unit',
    })
    ieee_block = 0
    other_rows = 0
    other_block_rows = 0
    for r in year_all[2024]:
        if r['reporting_standard'] == 'IEEE' and r['block_saidi_nomed'] == 'IEEE':
            ieee_block += 1
        if r['reporting_standard'] == 'OTHER':
            other_rows += 1
            if r['block_saidi_nomed'] == 'OTHER':
                other_block_rows += 1
    has_other = year_meta[2024]['has_other_block']
    checks.append({
        'check': '2024 standard-specific blocks are used for their own reporters',
        'observed': 'IEEE rows drawn from the IEEE block ' + str(ieee_block) + ' ; rows resolved as other standard ' + str(other_rows) + ' ; of those drawn from the other-standard block ' + str(other_block_rows) + ' ; other-standard block present ' + str(has_other),
        'expected': 'IEEE reporters are populated from the IEEE block, and when a distinct other-standard block exists its reporters are not read from the IEEE block',
        'passed': ieee_block > 0 and ((not has_other) or (other_rows > 0 and other_block_rows == other_rows)),
    })
    uids = set()
    for y in year_rows:
        uids.update(year_rows[y].keys())
    cohorts = []
    y1_map = {}
    diag = {}

    def bump(k):
        diag[k] = diag.get(k, 0) + 1

    for uid in sorted(uids):
        for Y in range(2016, 2025):
            recs = []
            ok = True
            for yy in (Y - 3, Y - 2, Y - 1, Y + 1):
                rec = year_rows.get(yy, {}).get(uid)
                if rec is None:
                    ok = False
                    break
                recs.append(rec)
            if not ok:
                bump('missing_four_year_window')
                continue
            r3, r2, r1, rp1 = recs
            stds = set(r['reporting_standard'] for r in recs)
            if len(stds) != 1:
                bump('reporting_standard_changed_or_unresolved')
                continue
            std = next(iter(stds))
            if std not in ('IEEE', 'OTHER'):
                bump('reporting_standard_unresolved')
                continue
            st = r1['state']
            own = r1['ownership_code']
            if st is None:
                bump('state_missing')
                continue
            if own not in ALLOWED_OWNERSHIP:
                bump('ownership_not_allowed')
                continue
            cust = r1['customers']
            if (not math.isfinite(cust)) or cust < 1000:
                bump('customers_below_1000_or_missing')
                continue
            lag1 = r1['saidi_nomed']
            lag2 = r2['saidi_nomed']
            lag3 = r3['saidi_nomed']
            y1 = rp1['saidi_nomed']
            if not all(math.isfinite(x) for x in (lag1, lag2, lag3)):
                bump('saidi_nomed_history_missing')
                continue
            if not math.isfinite(y1):
                bump('saidi_nomed_target_missing')
                continue
            base = float(np.median([lag1, lag2, lag3]))
            if not math.isfinite(base):
                bump('baseline_median_missing')
                continue
            if lag3 != 0 and math.isfinite(lag3):
                trend = (lag1 - lag3) / lag3
            else:
                trend = np.nan
            target = 1 if y1 >= 1.5 * base else 0
            cid = uid + '-' + str(Y)
            y1_map[cid] = y1
            cohorts.append({
                ID_COL: cid,
                TIME_COL: str(Y) + '-12-31T00:00:00.000',
                'utility_id': uid,
                'forecast_year': Y,
                'saidi_nomed_lag1': lag1,
                'saidi_nomed_lag2': lag2,
                'saidi_nomed_lag3': lag3,
                'saidi_nomed_trend': trend,
                'saidi_with_med_lag1': r1['saidi_with_med'],
                'saifi_nomed_lag1': r1['saifi_nomed'],
                'saifi_with_med_lag1': r1['saifi_with_med'],
                'caidi_nomed_lag1': r1['caidi_nomed'],
                'customers_lag1': cust,
                'reporting_standard': std,
                'ownership_code': own,
                'state': st,
                TARGET: target,
            })
    if not cohorts:
        raise AssertionError('No eligible utility-cohort rows. Diagnostics ' + str(diag)[:3000])
    df = pd.DataFrame(cohorts)
    df = df.sort_values([TIME_COL, 'utility_id', 'forecast_year']).reset_index(drop=True)
    train = df[df[TIME_COL] < SPLIT_EVAL_START]
    ev = df[(df[TIME_COL] >= SPLIT_EVAL_START) & (df[TIME_COL] < SPLIT_HOLDOUT_START)]
    hold = df[df[TIME_COL] >= SPLIT_HOLDOUT_START]
    if len(train) == 0 or len(ev) == 0 or len(hold) == 0:
        raise AssertionError('Empty split after eligibility. Counts ' + str((len(train), len(ev), len(hold))) + '. Diagnostics ' + str(diag)[:2000])
    _write(train, output / 'public' / 'train.csv')
    _write(ev, output / 'public' / 'eval.csv')
    _write(hold, output / 'private' / 'holdout.csv')
    split_counts = {'train': int(len(train)), 'eval': int(len(ev)), 'holdout': int(len(hold))}
    split_pos = {'train': int(train[TARGET].sum()), 'eval': int(ev[TARGET].sum()), 'holdout': int(hold[TARGET].sum())}
    checks.append({
        'check': 'cohort window completeness and eligibility filters',
        'observed': 'cohorts ' + str(len(df)) + ' ; diagnostics ' + str(diag),
        'expected': 'eligible cohorts carry one reliability record in each of calendar years Y minus 3, Y minus 2, Y minus 1 and Y plus 1, one consistent reporting standard, an allowed ownership category and at least 1000 lag-one customers',
        'passed': len(df) > 0,
    })
    checks.append({
        'check': 'deterministic chronological public and private split counts',
        'observed': str(split_counts) + ' positives ' + str(split_pos),
        'expected': 'nonempty train before 2020, eval from 2020 through 2021 and holdout from 2022 onward, each carrying positive labels',
        'passed': all(v > 0 for v in split_counts.values()) and all(v > 0 for v in split_pos.values()),
    })
    checks.append({
        'check': 'split boundaries are half-open and non-overlapping',
        'observed': 'train max ' + str(train[TIME_COL].max()) + ' ; eval min ' + str(ev[TIME_COL].min()) + ' ; eval max ' + str(ev[TIME_COL].max()) + ' ; holdout min ' + str(hold[TIME_COL].min()),
        'expected': 'train strictly below 2020-01-01, eval from 2020-01-01 below 2022-01-01, holdout at or after 2022-01-01',
        'passed': str(train[TIME_COL].max()) < SPLIT_EVAL_START and str(ev[TIME_COL].min()) >= SPLIT_EVAL_START and str(ev[TIME_COL].max()) < SPLIT_HOLDOUT_START and str(hold[TIME_COL].min()) >= SPLIT_HOLDOUT_START,
    })
    ids_dup = int(df[ID_COL].duplicated().sum())
    checks.append({
        'check': 'stable utility-cohort identifiers are unique',
        'observed': 'duplicate identifiers ' + str(ids_dup),
        'expected': 'zero duplicate utility cohort identifiers across all cohorts',
        'passed': ids_dup == 0,
    })
    tvals = sorted(set(int(x) for x in df[TARGET].unique()))
    checks.append({
        'check': 'binary target has no missing or unknown outcomes converted to a label',
        'observed': 'target values ' + str(tvals) + ' ; missing ' + str(int(df[TARGET].isna().sum())),
        'expected': 'only 0 and 1, with utilities lacking a CY Y plus 1 SAIDI value dropped rather than coded negative',
        'passed': set(tvals).issubset({0, 1}) and int(df[TARGET].isna().sum()) == 0,
    })
    cov = {c: int(df[c].notna().sum()) for c in FEATURES}
    empty = [c for c in FEATURES if cov[c] == 0]
    checks.append({
        'check': 'declared features have nonempty parsed coverage',
        'observed': str(cov),
        'expected': 'every declared feature column has at least one parsed value; missingness stays missing and is never zero filled',
        'passed': len(empty) == 0,
    })
    checks.append({
        'check': 'SAIDI trend contains no infinite values',
        'observed': 'infinite trend values ' + str(int(np.isinf(df['saidi_nomed_trend']).sum())) + ' ; missing ' + str(int(df['saidi_nomed_trend'].isna().sum())),
        'expected': 'trend finite when the lag-three SAIDI is nonzero and missing when it is zero, with no interpolation',
        'passed': int(np.isinf(df['saidi_nomed_trend']).sum()) == 0,
    })
    std_vals = sorted(str(x) for x in df['reporting_standard'].unique())
    checks.append({
        'check': 'reporting standard is resolved and consistent across the four filings of each cohort',
        'observed': 'resolved values ' + str(std_vals) + ' counts ' + str({k: int(v) for k, v in df['reporting_standard'].value_counts().items()}),
        'expected': 'only IEEE or OTHER after requiring one consistent standard across the four annual reliability filings',
        'passed': set(std_vals).issubset({'IEEE', 'OTHER'}),
    })
    checks.append({
        'check': 'ownership and state categories are resolved inside the shipped cohorts',
        'observed': 'ownership ' + str(sorted(str(x) for x in df['ownership_code'].unique())) + ' ; distinct states ' + str(int(df['state'].nunique())) + ' ; missing state ' + str(int(df['state'].isna().sum())),
        'expected': 'ownership among the six declared categories and a nonmissing two letter state code',
        'passed': set(df['ownership_code'].unique()).issubset(ALLOWED_OWNERSHIP) and int(df['state'].isna().sum()) == 0,
    })
    checks.append({
        'check': 'customer scale rule uses the lag-one filing only',
        'observed': 'minimum lag-one customers ' + str(float(df['customers_lag1'].min())),
        'expected': 'at least 1000 customers reported in calendar year Y minus 1',
        'passed': float(df['customers_lag1'].min()) >= 1000,
    })
    max_off = max(FEATURE_OFFSETS.values())
    checks.append({
        'check': 'feature years sit behind each forecast year while the label year sits ahead',
        'observed': 'maximum feature-year offset ' + str(max_off) + ' ; target-year offset 1',
        'expected': 'all features taken at or before calendar year Y minus 1 and the label taken only from calendar year Y plus 1',
        'passed': max_off <= -1,
    })
    forbidden = [c for c in df.columns if c != TARGET and ('rate' in c.lower() or 'target' in c.lower() or 'label' in c.lower())]
    checks.append({
        'check': 'no label-derived group rates or auxiliary answer files are shipped',
        'observed': 'forbidden derived columns ' + str(forbidden) + ' ; written files train.csv eval.csv holdout.csv',
        'expected': 'no eval or holdout group rates and no fixture labels or answer scaffolds',
        'passed': len(forbidden) == 0,
    })
    rule_bad = 0
    rule_n = 0
    for _idx, row in df.iterrows():
        y1 = y1_map.get(row[ID_COL])
        base = float(np.median([row['saidi_nomed_lag1'], row['saidi_nomed_lag2'], row['saidi_nomed_lag3']]))
        exp = 1 if (y1 is not None and math.isfinite(float(y1)) and float(y1) >= 1.5 * base) else 0
        rule_n += 1
        if int(row[TARGET]) != exp:
            rule_bad += 1
    checks.append({
        'check': 'shipped labels reproduce the frozen external rule on each stored row',
        'observed': 'rows checked ' + str(rule_n) + ' ; mismatches ' + str(rule_bad) + ' ; threshold ratio 1.5 applied to the median of the three stored lag values',
        'expected': 'one when calendar year Y plus 1 SAIDI excluding major event days is at or above 1.5 times the median of the stored Y minus 3, Y minus 2 and Y minus 1 values, zero otherwise',
        'passed': rule_n > 0 and rule_bad == 0,
        'kind': 'unit',
    })
    failed = [c for c in checks if not c['passed']]
    if failed:
        raise AssertionError('Source checks failed: ' + str(failed)[:4000])
    return checks
