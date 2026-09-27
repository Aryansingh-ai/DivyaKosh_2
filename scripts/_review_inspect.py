import csv, re, datetime
from pathlib import Path

rc_csv = Path(r'c:\Users\aryan\OneDrive\Documents\PROJECTS\DivyaKosh\data2\data\real_currency\manifest.csv')
with open(rc_csv, 'r', encoding='utf-8') as f:
    all_rows = {r['image_path']: r for r in csv.DictReader(f)}

no_cand_rows = [r for r in all_rows.values()
                if not re.search(r'CAND_', r.get('review_notes',''))
                and r['physical_note_id'] == 'UNKNOWN']
print(f'UNKNOWN no-CAND: {len(no_cand_rows)}')

uuid_re = re.compile(r'^[0-9a-f]{8}-[0-9a-f]{4}', re.I)
wa_re   = re.compile(r'WhatsApp Image (\d{4}-\d{2}-\d{2}) at (\d{2})\.(\d{2})\.(\d{2})')
img_re  = re.compile(r'IMG(\d{8})')

uuid_rows, wa_rows, img_rows, other_rows = [], [], [], []
for r in no_cand_rows:
    fname = Path(r['image_path']).name
    if uuid_re.match(fname):
        uuid_rows.append(r)
    elif wa_re.search(fname):
        wa_rows.append(r)
    elif img_re.search(fname):
        img_rows.append(r)
    else:
        other_rows.append(r)

print(f'UUID: {len(uuid_rows)}  WA: {len(wa_rows)}  IMG: {len(img_rows)}  Other: {len(other_rows)}')

print('\n--- WhatsApp rows by denom+time ---')
wa_by_denom = {}
for r in wa_rows:
    d = r['denomination']
    fname = Path(r['image_path']).name
    m = wa_re.search(fname)
    if m:
        dt = datetime.datetime(int(m.group(1)[:4]), int(m.group(1)[5:7]), int(m.group(1)[8:10]),
                               int(m.group(2)), int(m.group(3)), int(m.group(4)))
        wa_by_denom.setdefault(d, []).append((dt, fname, r['image_path']))
for d, items in sorted(wa_by_denom.items()):
    items.sort()
    print(f'  denom={d} n={len(items)} first={items[0][0]} last={items[-1][0]}')
    gaps = [(items[i+1][0]-items[i][0]).total_seconds() for i in range(len(items)-1)]
    large_gaps = [(i, g) for i, g in enumerate(gaps) if g > 300]
    print(f'    Gaps >5min at positions: {large_gaps}')

print('\n--- UUID rows by denom ---')
uuid_by_denom = {}
for r in uuid_rows:
    uuid_by_denom.setdefault(r['denomination'], []).append(r['image_path'])
for d, items in sorted(uuid_by_denom.items()):
    print(f'  denom={d} n={len(items)}')

print('\n--- IMG rows ---')
for r in img_rows:
    print(' ', r['image_path'])

print('\n--- Other rows ---')
for r in other_rows:
    print(' ', r['image_path'], r.get('review_notes','')[:60])
