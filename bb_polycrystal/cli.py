"""Batch CLI uses exactly the same pipeline as the desktop application."""
import argparse
from collections import Counter
import csv
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
from .core import Config, Engine, save_result


def _result_names(files):
    """Distinct, deterministic folder names, including same-stem input formats."""
    stems = Counter(f.stem.casefold() for f in files)
    candidates = [f.stem if stems[f.stem.casefold()] == 1
                  else f'{f.stem}_{f.suffix.lower().lstrip(".")}' for f in files]
    counts = Counter(name.casefold() for name in candidates)
    names = []
    used = set()
    for f, candidate in zip(files, candidates):
        if counts[candidate.casefold()] > 1:
            digest = hashlib.sha256(str(f.resolve()).encode('utf-8')).hexdigest()[:8]
            candidate = f'{candidate}_{digest}'
        name = candidate
        index = 2
        while name.casefold() in used:
            name = f'{candidate}_{index}'
            index += 1
        names.append(name)
        used.add(name.casefold())
    return names


def _available_output_dir(requested, overwrite=False):
    """Preserve the entire old run, including summaries, unless opted in."""
    requested = Path(requested)
    if overwrite or not requested.exists():
        return requested
    stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')
    candidate = requested.with_name(f'{requested.name}_{stamp}')
    index = 2
    while candidate.exists():
        candidate = requested.with_name(f'{requested.name}_{stamp}_{index}')
        index += 1
    return candidate


def main():
    p=argparse.ArgumentParser(description='BB PolyCrystal — пакетная сегментация MobileSAM на CPU')
    p.add_argument('input',type=Path)
    p.add_argument('--output',type=Path,default=Path('results'))
    p.add_argument('--overwrite',action='store_true',help='Разрешить перезапись результатов в существующем выходном каталоге')
    p.add_argument('--checkpoint',type=Path,default=Path('models/mobile_sam.pt'))
    p.add_argument('--crop-bottom',type=int,default=0,help='Число строк снизу. 0 = без обрезки, проверьте служебную полосу!')
    p.add_argument('--crop-manifest',type=Path,help='JSON filename → число удаляемых строк')
    p.add_argument('--points',type=int,default=24)
    p.add_argument('--keep-border', action='store_true', help='Сохранять объекты у края')
    p.add_argument('--border-margin', type=int, default=1, help='Отступ от рамки рабочей области, px')
    p.add_argument('--max-area-fraction', type=float, default=.85, help='Макс. доля поля для одной маски; 1 отключает фильтр')
    p.add_argument('--clahe',type=float,default=2)
    p.add_argument('--denoise',action='store_true',help='Медианный фильтр 3 × 3 перед CLAHE')
    p.add_argument('--min-area',type=int,default=16,help='Только малые артефакты, px²; 0 отключает фильтр')
    p.add_argument('--scale',type=float,default=None,help='мкм/пиксель; только подтверждённая калибровка')
    a=p.parse_args()
    files=sorted(f for f in a.input.iterdir() if f.is_file() and f.suffix.lower() in ['.png','.tif','.tiff','.jpg','.jpeg','.bmp','.webp']) if a.input.is_dir() else [a.input]
    names=_result_names(files)
    manifest=json.loads(a.crop_manifest.read_text()) if a.crop_manifest else {}
    requested_output=a.output
    while True:
        a.output=_available_output_dir(requested_output,a.overwrite)
        try:
            # Reserve before slow model loading/inference, so two CLI runs
            # cannot select the same still-empty destination.
            a.output.mkdir(parents=True,exist_ok=a.overwrite)
            break
        except FileExistsError:
            if a.overwrite:
                raise
    e=Engine(a.checkpoint)
    rows=[];failed=[]
    for f,name in zip(files,names):
        try:
            c=Config(crop_bottom=int(manifest.get(f.name,a.crop_bottom)),points_per_side=a.points,exclude_border=not a.keep_border,border_margin=a.border_margin,max_area_fraction=a.max_area_fraction,clahe_clip=a.clahe,denoise=a.denoise,min_area=a.min_area,scale_um_per_px=a.scale)
            r=e.segment(f,c); save_result(r,a.output/name)
            row={'file':f.name,**r.summary};rows.append(row)
            print(json.dumps(row,ensure_ascii=False),flush=True)
        except Exception as ex:
            failed.append({'file':f.name,'error':str(ex)})
            print(f'ОШИБКА {f.name}: {ex}',flush=True)
    a.output.mkdir(parents=True,exist_ok=True)
    (a.output/'batch_summary.json').write_text(json.dumps({'results':rows,'failed':failed},ensure_ascii=False,indent=2))
    if rows:
        cols=['file','raw_candidates','raw_components','excluded_field_masks','excluded_border','excluded_small','count','grain_count','coverage_fraction','inference_seconds']
        with (a.output/'batch_summary.csv').open('w',encoding='utf-8-sig',newline='') as out:
            w=csv.DictWriter(out,fieldnames=cols,extrasaction='ignore');w.writeheader();w.writerows(rows)
    print(f'Результаты: {a.output}',flush=True)
    if failed:raise SystemExit(1)

if __name__=='__main__':main()
