"""Image pipeline. No training, learned thresholds, or proprietary segmentation methods.

MobileSAM's official AutomaticMaskGenerator does inference and proposal NMS.
OpenCV CLAHE, skimage connected components and regionprops do preprocessing
and measurements. Sobel is a diagnostic panel, never a probability map.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
import csv
import hashlib
import json
import time
import warnings
import shutil

import cv2
import numpy as np
from PIL import Image
from scipy.optimize import linear_sum_assignment
from skimage import measure
from skimage.filters import sobel
from skimage.segmentation import find_boundaries


@dataclass
class Config:
    crop_bottom: int = 0
    clahe_clip: float = 2.0
    denoise: bool = False
    points_per_side: int = 24
    pred_iou_thresh: float = 0.88
    stability_score_thresh: float = 0.95
    box_nms_thresh: float = 0.3
    min_area: int = 0
    exclude_border: bool = True
    scale_um_per_px: float | None = None
    max_area_fraction: float = 0.85
    border_margin: int = 1

    def validate(self, shape):
        if not 0 <= self.crop_bottom < shape[0] - 15:
            raise ValueError("Рабочая область должна содержать не менее 16 строк")
        if not 4 <= self.points_per_side <= 64:
            raise ValueError("Число точек на сторону должно быть от 4 до 64")
        if not 0 < self.max_area_fraction <= 1:
            raise ValueError("Максимальная доля поля должна быть в (0, 1]")
        if self.border_margin < 0 or self.border_margin * 2 >= min(shape[0] - self.crop_bottom, shape[1]):
            raise ValueError("Отступ края должен быть меньше половины рабочей области")
        if self.min_area < 0 or self.clahe_clip < 0:
            raise ValueError("Параметры площади и CLAHE не могут быть отрицательными")
        if not (0 <= self.pred_iou_thresh <= 1 and 0 <= self.stability_score_thresh <= 1 and 0 <= self.box_nms_thresh <= 1):
            raise ValueError("Пороги confidence и stability должны лежать в [0, 1]")
        if self.scale_um_per_px is not None and (not np.isfinite(self.scale_um_per_px) or self.scale_um_per_px <= 0):
            raise ValueError("Масштаб должен быть положительным числом мкм/пиксель")


@dataclass
class Result:
    source: Path
    config: Config
    original: np.ndarray
    preprocessed: np.ndarray
    edge_map: np.ndarray
    raw_labels: np.ndarray
    labels: np.ndarray
    records: list[dict] = field(default_factory=list)
    summary: dict = field(default_factory=dict)
    classes: dict[int, str] = field(default_factory=dict)
    scores: dict[int, float | None] = field(default_factory=dict)
    edit_log: list[dict] = field(default_factory=list)
    proposal_masks: np.ndarray | None = None
    proposal_ids: dict[int, int] = field(default_factory=dict)
    original_border_ids: list[int] = field(default_factory=list)
    proposal_metadata: list[dict] = field(default_factory=list)
    proposal_archive: Path | None = None


def load_image(path: str | Path) -> np.ndarray:
    """PNG/JPEG/TIFF, palette, 8/16 bit and finite floating grayscale images.

    Non-8-bit grayscale is linearly mapped using its observed min/max. This is
    recorded in metadata, and never interpreted as a physical intensity scale.
    """
    with Image.open(path) as im:
        if im.mode in ("P", "RGB", "RGBA", "L", "1", "CMYK"):
            return np.asarray(im.convert("RGB"), dtype=np.uint8).copy()
        a = np.asarray(im).astype(np.float64)
    if not np.all(np.isfinite(a)):
        raise ValueError("Изображение содержит NaN или бесконечные значения")
    lo, hi = float(a.min()), float(a.max())
    a = np.zeros_like(a, dtype=np.uint8) if hi == lo else np.rint((a-lo)/(hi-lo)*255).astype(np.uint8)
    if a.ndim == 2:
        a = np.repeat(a[..., None], 3, axis=2)
    if a.ndim != 3 or a.shape[2] < 3:
        raise ValueError("Неподдерживаемое число каналов")
    return np.ascontiguousarray(a[..., :3])


def preprocess(rgb: np.ndarray, config: Config) -> np.ndarray:
    config.validate(rgb.shape)
    a = rgb[:rgb.shape[0]-config.crop_bottom].copy() if config.crop_bottom else rgb.copy()
    gray = cv2.cvtColor(a, cv2.COLOR_RGB2GRAY)
    if config.denoise:
        gray = cv2.medianBlur(gray, 3)
    if config.clahe_clip > 0:
        gray = cv2.createCLAHE(clipLimit=config.clahe_clip, tileGridSize=(8, 8)).apply(gray)
    return np.repeat(gray[..., None], 3, axis=2)


def proposals_to_labels(proposals: list[dict], shape: tuple[int, int], with_proposals=False):
    """Resolve overlap by higher model predicted IoU; split disconnected parts.

    These are bookkeeping decisions, not new grain-boundary detection. Original
    proposal bitmasks remain available in proposals.npz for inspecting overlap.
    """
    labels = np.zeros(shape, np.int32)
    scores = {}
    proposal_ids = {}
    original_border_ids = []
    next_id = 1
    for original_index,p in sorted(enumerate(proposals), key=lambda item: -float(item[1]["predicted_iou"])):
        pid=int(p.get("_proposal_index",original_index))
        mask=np.asarray(p["segmentation"],bool)
        border=bool(mask[0].any() or mask[-1].any() or mask[:,0].any() or mask[:,-1].any())
        free = mask & (labels == 0)
        cc = measure.label(free, connectivity=1)
        for r in measure.regionprops(cc):
            labels[cc == r.label] = next_id
            scores[next_id] = float(p["predicted_iou"])
            proposal_ids[next_id] = pid
            if border: original_border_ids.append(next_id)
            next_id += 1
    return (labels,scores,proposal_ids,original_border_ids) if with_proposals else (labels, scores)


def postprocess(labels: np.ndarray, config: Config, proposal_ids=None, original_border_ids=()):
    """Largest connected component per proposal, ROI border zone, explicit noise area.

    Keeping the largest connected component is standard component filtering. It
    prevents overlap leftovers of ONE model proposal becoming dozens of grains.
    No ellipse/size-distribution or mean-area plausibility rules are applied.
    """
    out = labels.copy()
    if proposal_ids:
        groups={}
        sizes=np.bincount(out.ravel())
        for label,pid in proposal_ids.items():groups.setdefault(pid,[]).append(label)
        for ids in groups.values():
            largest=max(ids,key=lambda i:sizes[i])
            out[np.isin(out,[i for i in ids if i!=largest])]=0
    margin = config.border_margin + 1
    border_ids = (set(np.unique(np.concatenate((out[:margin].ravel(), out[-margin:].ravel(),
                  out[:, :margin].ravel(), out[:, -margin:].ravel())))) | set(original_border_ids)) - {0}
    removed_border = removed_small = 0
    for r in measure.regionprops(out):
        if config.exclude_border and r.label in border_ids:
            out[out == r.label] = 0
            removed_border += 1
        elif r.area < config.min_area:
            out[out == r.label] = 0
            removed_small += 1
    return out, removed_border, removed_small


def touches_border(mask: np.ndarray, margin: int = 0) -> bool:
    """The ROI includes the new bottom edge after cropping the microscope strip."""
    width = margin + 1
    return bool(mask[:width].any() or mask[-width:].any() or
                mask[:, :width].any() or mask[:, -width:].any())


def process_proposals(proposals, shape, config):
    """Reject field masks independently of edge policy, before overlap assignment.

    Keep the largest ORIGINAL connected component first: a disconnected satellite
    at the edge must not condemn an interior object. Source border provenance is
    retained even when overlap removes its contact pixels. Full unmodified SAM
    bitmasks remain in proposals.npz, outside the mutually exclusive preview map.
    """
    candidates = []
    rejected_field = rejected_border = rejected_small = fragments = 0
    decisions = []
    for index, proposal in enumerate(proposals):
        mask = np.asarray(proposal['segmentation'], dtype=bool)
        if mask.shape != tuple(shape):
            raise ValueError("Размер предложения SAM не совпадает с рабочей областью")
        area = int(mask.sum())
        if not area:
            decisions.append(dict(index=index, reason="empty"))
            continue
        # A value of 1 disables this filter. The threshold is explicit and saved.
        if config.max_area_fraction < 1 and area / mask.size > config.max_area_fraction:
            rejected_field += 1
            decisions.append(dict(index=index, reason="field_mask", area=area))
            continue
        cc = measure.label(mask, connectivity=1)
        sizes = np.bincount(cc.ravel()); sizes[0] = 0
        mask = cc == int(sizes.argmax())
        fragments += int(np.count_nonzero(sizes)) - 1
        candidates.append(dict(proposal, segmentation=mask, _proposal_index=index))
    # Stage 04 retains edge and small masks. Degenerate field proposals have
    # already been rejected, so they cannot turn this diagnostic into one fill.
    raw, raw_scores = proposals_to_labels(candidates, shape)
    eligible = []
    for proposal in candidates:
        mask = proposal['segmentation']
        reason = None
        if config.exclude_border and touches_border(mask, config.border_margin):
            rejected_border += 1
            reason = "border"
        elif mask.sum() < config.min_area:
            rejected_small += 1
            reason = "small"
        if reason:
            decisions.append(dict(index=proposal['_proposal_index'], reason=reason, area=int(mask.sum())))
        else:
            eligible.append(proposal)
    resolved, scores, pids, _ = proposals_to_labels(eligible, shape, True)
    source_borders = {p['_proposal_index'] for p in eligible
                      if touches_border(p['segmentation'], config.border_margin)}
    original_borders = [ident for ident, pid in pids.items() if pid in source_borders]
    final, border, small = postprocess(resolved, config, pids, original_borders)
    stats = dict(raw_candidates=len(proposals), raw_components=len(raw_scores),
                 excluded_field_masks=rejected_field, excluded_border=rejected_border + border,
                 excluded_small=rejected_small + small, eligible_components=len(scores),
                 excluded_source_fragments=fragments,
                 excluded_overlap_fragments=len(scores) - len(set(pids.values())),
                 proposal_decisions=decisions, pipeline_version=2)
    return raw, final, scores, pids, original_borders, stats


def read_proposals(result: Result):
    if result.proposal_masks is not None:
        masks = result.proposal_masks
    elif result.proposal_archive and result.proposal_archive.is_file():
        with np.load(result.proposal_archive, allow_pickle=False) as data:
            masks = np.unpackbits(data['packed_masks'], axis=-1)[..., :int(data['shape'][-1])].astype(bool)
    else:
        raise ValueError("В сессии нет исходных масок SAM. Запустите обработку снимка заново.")
    metadata = {row['index']: row for row in result.proposal_metadata}
    return [dict(metadata.get(i, {}), segmentation=mask,
                 predicted_iou=metadata.get(i, {}).get('predicted_iou', 0.))
            for i, mask in enumerate(masks)]


def refilter_result(result: Result, config: Config):
    """Rebuild from archived model proposals; do not silently mix edited geometry."""
    config.validate(result.original.shape)
    raw, labels, scores, pids, borders, stats = process_proposals(
        read_proposals(result), result.labels.shape, config)
    # Retain user classes only when geometry is exactly unchanged.
    classes = {}
    old_by_proposal = {}
    for ident, pid in result.proposal_ids.items():
        old_by_proposal.setdefault(pid, []).append(ident)
    for ident, pid in pids.items():
        if not np.any(labels == ident):
            continue
        for old in old_by_proposal.get(pid, []):
            if np.array_equal(labels == ident, result.labels == old):
                classes[ident] = result.classes.get(old, "Зерно")
                break
    result.config = config
    result.raw_labels, result.labels = raw, labels
    result.scores, result.proposal_ids = scores, pids
    result.original_border_ids = borders
    result.classes = classes
    result.summary.update(stats)
    result.summary['reference_metrics'] = None
    return refresh_result(result)


class Engine:
    def __init__(self, checkpoint: Path):
        import torch
        from mobile_sam import sam_model_registry
        if not Path(checkpoint).is_file():
            raise FileNotFoundError(f"Нет весов модели: {checkpoint}. См. README.md")
        torch.set_num_threads(4)
        self.checkpoint = Path(checkpoint)
        self.checkpoint_sha256 = hashlib.sha256(self.checkpoint.read_bytes()).hexdigest()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", FutureWarning)
            self.model = sam_model_registry["vit_t"](checkpoint=str(checkpoint))
        self.model.to(device="cpu").eval()

    def segment(self, path, config: Config) -> Result:
        import torch
        from mobile_sam import SamAutomaticMaskGenerator
        original = load_image(path)
        pre = preprocess(original, config)
        edges = sobel(pre[..., 0].astype(np.float32)/255)
        generator = SamAutomaticMaskGenerator(
            self.model, points_per_side=config.points_per_side,
            points_per_batch=32, pred_iou_thresh=config.pred_iou_thresh,
            stability_score_thresh=config.stability_score_thresh,
            box_nms_thresh=config.box_nms_thresh, crop_n_layers=0, min_mask_region_area=0)
        start = time.perf_counter()
        with torch.inference_mode():
            proposals = generator.generate(pre)
        seconds = time.perf_counter()-start
        raw, labels, scores, pids, original_borders,stats = process_proposals(proposals,pre.shape[:2],config)
        masks = np.stack([p['segmentation'] for p in proposals]) if proposals else np.zeros((0,*pre.shape[:2]), bool)
        with Image.open(path) as im:
            source_mode = im.mode
        result = Result(Path(path), config, original, pre, edges, raw, labels,
                        summary={"model":"MobileSAM vit_t", "checkpoint_sha256":self.checkpoint_sha256,
                            **stats,
                            "inference_seconds":seconds,"reference_metrics":None,"device":"cpu",
                            "source_mode":source_mode,"source_sha256":hashlib.sha256(Path(path).read_bytes()).hexdigest(),
                            "intensity_mapping":"palette/RGB conversion to gray" if source_mode in ('P','RGB','RGBA','L') else "observed min-max to uint8",
                            "crop_coordinates_xyxy":[0,0,pre.shape[1],pre.shape[0]],
                            "nms_threshold":config.box_nms_thresh,"crop_n_layers":0,
                            "edge_map_type":"Sobel magnitude; diagnostic, not probability"},
                        scores=scores,proposal_masks=masks,proposal_ids=pids,original_border_ids=original_borders,
                        proposal_metadata=[dict(index=i,predicted_iou=float(p['predicted_iou']),
                            stability_score=float(p['stability_score']),area=int(p['area'])) for i,p in enumerate(proposals)])
        refresh_result(result)
        return result


def refresh_result(result: Result):
    """Measure current labels; edited mask classes are user supplied."""
    result.config.validate(result.original.shape)
    if result.labels.shape != result.preprocessed.shape[:2]:
        raise ValueError("Размер маски не совпадает с рабочей областью")
    raw_gray = cv2.cvtColor(result.original[:result.labels.shape[0]], cv2.COLOR_RGB2GRAY)
    records = []
    for r in measure.regionprops(result.labels):
        mask = result.labels == r.label
        boundary = find_boundaries(mask, mode="inner")
        intensities = raw_gray[boundary].astype(float)
        edges = result.edge_map[boundary]
        row = {"id":int(r.label),"class":result.classes.get(r.label,"Зерно"),
            "area_px2":float(r.area),"centroid_x_px":float(r.centroid[1]),"centroid_y_px":float(r.centroid[0]),
            "perimeter_px":float(r.perimeter_crofton),"major_axis_px":float(r.axis_major_length),
            "minor_axis_px":float(r.axis_minor_length),"eccentricity":float(r.eccentricity),
            "feret_max_px":float(r.feret_diameter_max),
            "boundary_intensity_mean":float(intensities.mean()) if len(intensities) else 0.,
            "boundary_intensity_std":float(intensities.std()) if len(intensities) else 0.,
            "edge_strength_mean":float(edges.mean()) if len(edges) else 0.,
            "edge_strength_std":float(edges.std()) if len(edges) else 0.,
            "touches_border":touches_border(mask, result.config.border_margin),
            "predicted_iou":result.scores.get(int(r.label))}
        scale = result.config.scale_um_per_px
        if scale:
            row['area_um2'] = row['area_px2']*scale**2
            for key in ('centroid_x','centroid_y','perimeter','major_axis','minor_axis','feret_max'):
                row[key+'_um'] = row[key+'_px']*scale
        records.append(row)
    result.records = records
    result.summary.update(count=len(records),grain_count=sum(r['class']=='Зерно' for r in records),
        coverage_fraction=float(np.mean(result.labels>0)),
        grain_coverage_fraction=float(np.mean(np.isin(result.labels,[r['id'] for r in records if r['class']=='Зерно']))))
    return result


CLASS_COLORS = {"Зерно": (101, 35, 43), "Артефакт": (172, 112, 65), "Неопределено": (92, 119, 117)}


def label_rgb(labels, classes=None):
    palette = np.array([(101,35,43),(190,108,126),(157,127,111),(110,140,131),
                        (122,104,142),(201,153,109),(169,82,95),(96,130,154),
                        (128,144,95),(216,153,158),(107,80,65),(165,133,166)], np.uint8)
    ids, inverse = np.unique(labels, return_inverse=True)
    colors = np.empty((len(ids), 3), np.uint8)
    for i, ident in enumerate(ids):
        if ident == 0:
            colors[i] = (248, 229, 215)
        elif classes is not None:
            colors[i] = CLASS_COLORS.get(classes.get(int(ident), "Зерно"), (92,119,117))
        else:
            colors[i] = palette[(int(ident) - 1) % len(palette)]
    return colors[inverse].reshape((*labels.shape, 3))


def draw_label_boundaries(image: np.ndarray, labels: np.ndarray) -> np.ndarray:
    """Light contour with a dark rim stays visible over both image and mask fill."""
    boundary = find_boundaries(labels, mode='inner')
    rim = cv2.dilate(boundary.astype(np.uint8), np.ones((3, 3), np.uint8)).astype(bool)
    image[rim] = (101, 35, 43)
    image[boundary] = (255, 240, 218)
    return image


def edge_preview(edge_map: np.ndarray) -> np.ndarray:
    """Stretch only the diagnostic display; measurements keep the original Sobel."""
    values = np.nan_to_num(np.asarray(edge_map, dtype=np.float32), nan=0., posinf=0., neginf=0.)
    if not np.any(values > 0):
        return np.zeros(values.shape, np.uint8)
    low, high = np.percentile(values, [50, 99.5])
    if high <= low:
        low, high = 0., float(values.max())
    return np.rint(np.clip((values - low) / (high - low), 0, 1) * 255).astype(np.uint8)


def stage_images(result: Result, color_by_class=False) -> dict[str,np.ndarray]:
    colors = label_rgb(result.labels, result.classes if color_by_class else None)
    overlay = result.original[:result.labels.shape[0]].copy()
    m = result.labels > 0
    overlay[m] = (.30*overlay[m]+.70*colors[m]).astype(np.uint8)
    draw_label_boundaries(overlay, result.labels)
    edge = edge_preview(result.edge_map)
    return {"Original":result.original,"Preprocessed":result.preprocessed,
        "Edge map":np.repeat(edge[...,None],3,axis=2),"Raw segmentation":label_rgb(result.raw_labels),
        "Postprocessed segmentation":colors,"Overlay on original":overlay}


def evaluate_labels(pred, ref, threshold=.5):
    """Full-ROI binary and instance metrics, with one-to-one IoU matching."""
    pred, ref = np.asarray(pred), np.asarray(ref)
    if pred.ndim!=2 or pred.shape!=ref.shape:
        raise ValueError("Эталон должен иметь тот же двумерный размер, что рабочая область")
    for a in (pred,ref):
        if not np.all(np.isfinite(a)) or np.any(a<0) or not np.all(a==np.floor(a)):
            raise ValueError("Маски должны содержать неотрицательные целые ID, 0 — фон")
    a,b=pred>0,ref>0
    inter=int(np.count_nonzero(a&b));union=int(np.count_nonzero(a|b))
    na,nb=int(a.sum()),int(b.sum())
    # Compress categorical IDs first: a label such as 1_000_000_000 must not
    # allocate a billion-bin histogram. Each pixel contributes once to the
    # joint table, rather than scanning the full image for each object pair.
    pvalues,pinverse=np.unique(pred.ravel(),return_inverse=True)
    rvalues,rinverse=np.unique(ref.ravel(),return_inverse=True)
    joint=np.bincount(pinverse*len(rvalues)+rinverse,
                      minlength=len(pvalues)*len(rvalues)).reshape(len(pvalues),len(rvalues))
    pforeground=pvalues>0;rforeground=rvalues>0
    pids=pvalues[pforeground];rids=rvalues[rforeground]
    intersections=joint[np.ix_(pforeground,rforeground)]
    pareas=joint.sum(axis=1)[pforeground]
    rareas=joint.sum(axis=0)[rforeground]
    unions=pareas[:,None]+rareas[None,:]-intersections
    iou=np.divide(intersections,unions,out=np.zeros(intersections.shape,dtype=float),where=unions>0)
    if iou.size:
        # Maximize valid match count first, overlap second.
        cost=-((iou>=threshold)*(min(iou.shape)+1)+iou)
        ii,jj=linear_sum_assignment(cost)
        hits=iou[ii,jj]>=threshold
        tp=int(hits.sum());mean_iou=float(iou[ii[hits],jj[hits]].mean()) if tp else 0.
    else:tp=0;mean_iou=0.
    fp,fn=len(pids)-tp,len(rids)-tp
    return dict(iou=inter/union if union else 1.,dice=2*inter/(na+nb) if na+nb else 1.,
        precision=inter/na if na else float(nb==0),recall=inter/nb if nb else float(na==0),
        object_tp=tp,object_fp=fp,object_fn=fn,object_f1=2*tp/(2*tp+fp+fn) if 2*tp+fp+fn else 1.,
        object_mean_matched_iou=mean_iou,object_iou_threshold=threshold,
        note="Binary metrics use foreground union; object F1 penalizes merges/splits. Reference must be independently annotated.")


def save_result(result: Result, out_dir) -> Path:
    out=Path(out_dir);out.mkdir(parents=True,exist_ok=True)
    refresh_result(result)
    for name,im in stage_images(result).items():
        Image.fromarray(im).save(out/(name.lower().replace(' ','_')+'.png'))
    Image.fromarray(result.labels.astype(np.int32),mode='I').save(out/'labels.tif')
    Image.fromarray(result.raw_labels.astype(np.int32),mode='I').save(out/'raw_labels.tif')
    class_names = ['Зерно', 'Артефакт', 'Неопределено']
    class_names += sorted(set(result.classes.values()) - set(class_names))
    class_ids = {name: i + 1 for i, name in enumerate(class_names)}
    class_map = np.zeros(result.labels.shape, np.uint16)
    for row in result.records:
        class_map[result.labels == row['id']] = class_ids[row['class']]
    Image.fromarray(class_map).save(out/'classes.tif')
    Image.fromarray(stage_images(result, color_by_class=True)['Overlay on original']).save(out/'overlay_by_class.png')
    np.savez_compressed(out/'session.npz',original=result.original,preprocessed=result.preprocessed,
                        edge_map=result.edge_map,raw_labels=result.raw_labels,labels=result.labels)
    if result.proposal_masks is not None:
        masks=result.proposal_masks
        np.savez_compressed(out/'proposals.npz',packed_masks=np.packbits(masks,axis=-1),shape=np.array(masks.shape))
        result.proposal_archive=(out/'proposals.npz').resolve()
        # Saved sessions keep the compressed source proposals on disk; batch GUI
        # must not retain hundreds of MB of dense proposal masks for each file.
        result.proposal_masks=None
    elif result.proposal_archive is not None and result.proposal_archive.is_file():
        if result.proposal_archive.resolve() != (out/'proposals.npz').resolve():
            shutil.copyfile(result.proposal_archive,out/'proposals.npz')
        result.proposal_archive=(out/'proposals.npz').resolve()
    with (out/'grains.csv').open('w',encoding='utf-8-sig',newline='') as f:
        fields=list(result.records[0]) if result.records else ['id','class','area_px2']
        writer=csv.DictWriter(f,fieldnames=fields);writer.writeheader();writer.writerows(result.records)
    meta=dict(schema_version=2,application="BB PolyCrystal",class_ids=class_ids,source=str(result.source),config=asdict(result.config),
              summary=result.summary,classes=result.classes,scores=result.scores,edit_log=result.edit_log,
              proposal_ids=result.proposal_ids,original_border_ids=result.original_border_ids,
              proposal_metadata=result.proposal_metadata)
    (out/'session.json').write_text(json.dumps(meta,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')
    return out


def load_result(folder) -> Result:
    folder=Path(folder)
    meta=json.loads((folder/'session.json').read_text(encoding='utf-8'))
    with np.load(folder/'session.npz',allow_pickle=False) as z:
        arrays={k:z[k].copy() for k in ('original','preprocessed','edge_map','raw_labels','labels')}
    config_data = dict(meta['config'])
    config_data.pop('device', None)  # Legacy sessions may store an obsolete GPU choice.
    r=Result(Path(meta['source']),Config(**config_data),**arrays,summary=meta['summary'],
             classes={int(k):v for k,v in meta['classes'].items()},scores={int(k):v for k,v in meta['scores'].items()},edit_log=meta.get('edit_log',[]),
             proposal_ids={int(k):v for k,v in meta.get('proposal_ids',{}).items()},original_border_ids=meta.get('original_border_ids',[]),
             proposal_metadata=meta.get('proposal_metadata',[]))
    refresh_result(r)
    if (folder/'proposals.npz').is_file():r.proposal_archive=(folder/'proposals.npz').resolve()
    return r
