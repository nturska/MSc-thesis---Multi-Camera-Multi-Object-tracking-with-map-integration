"""Hybrid-SORT — oficjalna implementacja (ymzis69/HybridSORT), opakowana do API dets→tracks."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import numpy as np

from .hybrid_sort import Hybrid_Sort


class HybridSORTTracker:
    """
    Wrapper wokół oficjalnego Hybrid_Sort.
    Wejście: dets (N,5) = [x1,y1,x2,y2,score] w współrzędnych obrazu.
    Wyjście: (M,5+) = [x1,y1,x2,y2,track_id, ...]
    """

    def __init__(self, cfg: dict[str, Any] | None = None):
        cfg = dict(cfg or {})
        det_thresh = float(cfg.get("det_thresh", cfg.get("track_high_thresh", 0.6)))
        track_thresh = float(cfg.get("track_thresh", det_thresh))
        args = SimpleNamespace(
            TCM_first_step=bool(cfg.get("TCM_first_step", True)),
            TCM_first_step_weight=float(cfg.get("hmiou_weight", cfg.get("TCM_first_step_weight", 1.0))),
            TCM_byte_step=bool(cfg.get("TCM_byte_step", True)),
            TCM_byte_step_weight=float(cfg.get("TCM_byte_step_weight", 1.0)),
            track_thresh=track_thresh,
        )
        self.tracker = Hybrid_Sort(
            args,
            det_thresh=det_thresh,
            max_age=int(cfg.get("track_buffer", cfg.get("max_age", 30))),
            min_hits=int(cfg.get("min_hits", 3)),
            iou_threshold=float(cfg.get("iou_threshold", 0.3)),
            delta_t=int(cfg.get("delta_t", 3)),
            asso_func=str(cfg.get("asso_func", "Height_Modulated_IoU")),
            inertia=float(cfg.get("inertia", 0.2)),
            use_byte=bool(cfg.get("use_byte", True)),
        )

    def update(self, dets: np.ndarray, frame: np.ndarray) -> np.ndarray:
        if dets is None or len(dets) == 0:
            dets = np.empty((0, 5), dtype=np.float32)
        else:
            dets = np.asarray(dets, dtype=np.float32)
            if dets.ndim == 1:
                dets = dets.reshape(1, -1)
            if dets.shape[1] > 5:
                dets = dets[:, :5]
        h, w = frame.shape[:2]
        # scale=1: detekcje już w przestrzeni oryginalnego obrazu
        out = self.tracker.update(dets, img_info=(h, w), img_size=(h, w))
        if out is None or len(out) == 0:
            return np.empty((0, 5), dtype=np.float32)
        return np.asarray(out, dtype=np.float32)
