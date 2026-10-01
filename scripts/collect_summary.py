from __future__ import annotations
import argparse,csv,json
from pathlib import Path
p=argparse.ArgumentParser(description='Collect all metrics.json files into one CSV')
p.add_argument('--root',required=True); p.add_argument('--output',default='summary.csv')
a=p.parse_args(); rows=[]
for m in sorted(Path(a.root).rglob('metrics.json')):
    try: rows.append(json.loads(m.read_text(encoding='utf-8')))
    except Exception: pass
if not rows: raise SystemExit('No metrics.json files found')
fields=[]
for r in rows:
    for k in r:
        if k not in fields: fields.append(k)
with open(a.output,'w',newline='',encoding='utf-8') as f:
    w=csv.DictWriter(f,fieldnames=fields); w.writeheader(); w.writerows(rows)
print(f'wrote {len(rows)} rows -> {a.output}')
