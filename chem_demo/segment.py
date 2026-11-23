"""Open-vocabulary detection (Grounding DINO) plus box-prompted masks (SAM 2.1) on the local GPU.

Models come from the Hugging Face hub on first use and are cached afterwards. The vocabulary is
the platform's object list; each detection keeps its prompt label, box, score and a binary mask.

    .venv-perception/Scripts/python.exe -X utf8 -m chem_demo.segment --image top.png --out <dir>
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Sequence

import cv2
import numpy as np

DETECTOR_ID = "IDEA-Research/grounding-dino-base"
# The hub download of SAM 2.1 stalled twice on this network; a local copy is used when present.
_LOCAL_SAM = Path(__file__).resolve().parents[1] / "calibration-data" / "chem-demo" / "models" / "sam2.1-hiera-large"
SEGMENTER_ID = str(_LOCAL_SAM) if (_LOCAL_SAM / "config.json").exists() else "facebook/sam2.1-hiera-large"

# Prompt label -> object role on the platform. Grounding DINO matches lower-case phrases ending in '.'.
VOCABULARY = {
    "glass beaker": "beaker",
    "glass bottle": "bottle",
    "plastic dropper pipette": "dropper",
    "glass stirring rod": "rod",
    "test tube rack": "rack",
    "electronic balance": "balance",
}
MAX_BOX_SHARE = {"rack": 0.25, "balance": 0.3}  # largest plausible box as a share of the image; default 0.2
COLORS = {"beaker": (255, 160, 0), "bottle": (0, 200, 255), "dropper": (255, 0, 255), "rod": (0, 255, 0),
          "rack": (0, 120, 255), "balance": (180, 180, 180)}


@dataclass
class Detection:
    label: str
    role: str
    score: float
    box: list[float]  # x0, y0, x1, y1 in pixels
    mask: np.ndarray = field(repr=False)
    mask_score: float = 0.0

    def summary(self) -> dict:
        ys, xs = np.nonzero(self.mask)
        centroid = [float(xs.mean()), float(ys.mean())] if len(xs) else None
        return {"label": self.label, "role": self.role, "score": round(self.score, 3),
                "mask_score": round(self.mask_score, 3), "box": [round(v, 1) for v in self.box],
                "mask_area": int(self.mask.sum()), "mask_centroid": centroid}


class Segmenter:
    def __init__(self, device: str = "cuda"):
        import torch
        from transformers import (AutoModelForZeroShotObjectDetection, AutoProcessor, Sam2Model,
                                  Sam2Processor)
        self.torch, self.device = torch, device
        self.det_processor = AutoProcessor.from_pretrained(DETECTOR_ID)
        self.detector = AutoModelForZeroShotObjectDetection.from_pretrained(DETECTOR_ID).to(device).eval()
        self.seg_processor = Sam2Processor.from_pretrained(SEGMENTER_ID)
        self.segmenter = Sam2Model.from_pretrained(SEGMENTER_ID).to(device).eval()

    def detect(self, image_bgr: np.ndarray, vocabulary: dict[str, str] = VOCABULARY,
               box_threshold: float = 0.3, text_threshold: float = 0.25,
               size_filter: bool = True) -> list[Detection]:
        """size_filter drops boxes larger than MAX_BOX_SHARE of the image (false hits on whole arms in
        the full top view); turn it off for crops and close-up wrist views, where a target is big."""
        from PIL import Image
        torch = self.torch
        image = Image.fromarray(cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB))
        # One query per class: with all phrases in one prompt, Grounding DINO merged labels
        # ("glass beaker glass", "plastic dropper pipette glass stirring") on the 2026-11-20 frames.
        width, height = image.size
        keep = []
        for label, role in vocabulary.items():
            inputs = self.det_processor(images=image, text=f"{label}.", return_tensors="pt").to(self.device)
            with torch.no_grad():
                outputs = self.detector(**inputs)
            result = self.det_processor.post_process_grounded_object_detection(
                outputs, inputs.input_ids, threshold=box_threshold, text_threshold=text_threshold,
                target_sizes=[(height, width)])[0]
            for box, score in zip(result["boxes"].cpu().numpy().tolist(), result["scores"].cpu().numpy().tolist()):
                share = (box[2] - box[0]) * (box[3] - box[1]) / (width * height)
                if not size_filter or share <= MAX_BOX_SHARE.get(role, 0.2):  # whole-arm boxes are false hits
                    keep.append((label, role, box, score))
        keep = _suppress(keep)
        if not keep:
            return []
        masks, mask_scores = self.masks(image, [k[2] for k in keep])
        return [Detection(label, role, float(score), [float(v) for v in box], mask, float(ms))
                for (label, role, box, score), mask, ms in zip(keep, masks, mask_scores)]

    def masks(self, image, boxes: list[list[float]]) -> tuple[list[np.ndarray], list[float]]:
        torch = self.torch
        inputs = self.seg_processor(images=image, input_boxes=[boxes], return_tensors="pt").to(self.device)
        with torch.no_grad():
            outputs = self.segmenter(**inputs, multimask_output=False)
        masks = self.seg_processor.post_process_masks(outputs.pred_masks.cpu(), inputs["original_sizes"])[0]
        masks = masks[:, 0].numpy() > 0.5 if masks.ndim == 4 else masks.numpy() > 0.5
        scores = outputs.iou_scores.cpu().numpy().reshape(len(boxes), -1)[:, 0].tolist()
        return [m.astype(np.uint8) for m in masks], scores


def _iou(a: Sequence[float], b: Sequence[float]) -> float:
    x0, y0, x1, y1 = max(a[0], b[0]), max(a[1], b[1]), min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, x1 - x0) * max(0.0, y1 - y0)
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


def _suppress(items: list[tuple], iou_limit: float = 0.6, cross_limit: float = 0.85) -> list[tuple]:
    """Non-maximum suppression: within a role at IoU 0.6, and across roles (one object answered to
    two phrases, e.g. the bottle also scored as a beaker) at IoU 0.85, keeping the higher score."""
    kept: list[tuple] = []
    for item in sorted(items, key=lambda i: -i[3]):
        if all(_iou(item[2], k[2]) < (iou_limit if item[1] == k[1] else cross_limit) for k in kept):
            kept.append(item)
    return kept


def draw(image: np.ndarray, detections: list[Detection]) -> np.ndarray:
    view = image.copy()
    overlay = view.copy()
    for d in detections:
        overlay[d.mask.astype(bool)] = COLORS.get(d.role, (255, 255, 255))
    view = cv2.addWeighted(overlay, 0.4, view, 0.6, 0)
    for d in detections:
        color = COLORS.get(d.role, (255, 255, 255))
        x0, y0, x1, y1 = map(int, d.box)
        cv2.rectangle(view, (x0, y0), (x1, y1), color, 2)
        cv2.putText(view, f"{d.role} {d.score:.2f}", (x0, max(18, y0 - 6)), cv2.FONT_HERSHEY_SIMPLEX,
                    0.6, color, 2)
    return view


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="开放词表检测 + SAM2 分割（离线，单张或多张图）")
    parser.add_argument("--image", type=Path, nargs="+", required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--box-threshold", type=float, default=0.3)
    args = parser.parse_args(argv)
    args.out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    segmenter = Segmenter()
    print(f"models loaded in {time.time() - t0:.1f} s")
    for path in args.image:
        image = cv2.imread(str(path))
        t1 = time.time()
        detections = segmenter.detect(image, box_threshold=args.box_threshold)
        elapsed = time.time() - t1
        cv2.imwrite(str(args.out / f"{path.stem}_seg.jpg"), draw(image, detections))
        for i, d in enumerate(detections):
            cv2.imwrite(str(args.out / f"{path.stem}_mask{i:02d}_{d.role}.png"), d.mask * 255)
        report = {"image": str(path), "seconds": round(elapsed, 2), "detections": [d.summary() for d in detections]}
        (args.out / f"{path.stem}_seg.json").write_text(json.dumps(report, ensure_ascii=False, indent=2),
                                                        encoding="utf-8")
        print(json.dumps({"image": path.name, "seconds": report["seconds"],
                          "found": [f"{d.role}:{d.score:.2f}" for d in detections]}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
