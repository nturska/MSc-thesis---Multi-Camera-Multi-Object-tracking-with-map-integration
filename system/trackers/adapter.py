"""
Uniwersalny adapter SCT: detekcja (Ultralytics predict) → tracker.update().

ByteTrack / BoT-SORT / Deep_OC-SORT / TrackTrack → klasy z `ultralytics.trackers`.
Hybrid-SORT / BoostTrack++ → lokalne `system/trackers/`.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from ultralytics.utils import IterableSimpleNamespace

from src.config import get_sct_tracker_cfg, resolve_tracker_source


class DetectionResults:
    """Minimalny obiekt Results-like wymagany przez Ultralytics BYTETracker / BOTSORT / …"""

    def __init__(self, xyxy: np.ndarray, conf: np.ndarray, cls: np.ndarray):
        self.xyxy = np.asarray(xyxy, dtype=np.float32).reshape(-1, 4)
        self.conf = np.asarray(conf, dtype=np.float32).reshape(-1)
        self.cls = np.asarray(cls, dtype=np.float32).reshape(-1)

    def __len__(self) -> int:
        return int(self.conf.shape[0])

    def __getitem__(self, idx) -> DetectionResults:
        return DetectionResults(self.xyxy[idx], self.conf[idx], self.cls[idx])

    @property
    def xywh(self) -> np.ndarray:
        x1, y1, x2, y2 = self.xyxy.T
        return np.stack([(x1 + x2) / 2.0, (y1 + y2) / 2.0, (x2 - x1), (y2 - y1)], axis=1).astype(np.float32)


def _ultralytics_args(cfg: dict[str, Any], device=None) -> IterableSimpleNamespace:
    """Buduje Namespace parametrów zgodny z konstruktorami Ultralytics trackerów."""
    return IterableSimpleNamespace(
        track_high_thresh=float(cfg.get("track_high_thresh", 0.25)),
        track_low_thresh=float(cfg.get("track_low_thresh", 0.1)),
        new_track_thresh=float(cfg.get("new_track_thresh", 0.3)),
        track_buffer=int(cfg.get("track_buffer", 30)),
        match_thresh=float(cfg.get("match_thresh", 0.8)),
        fuse_score=bool(cfg.get("fuse_score", True)),
        gmc_method=str(cfg.get("gmc_method", "sparseOptFlow")),
        proximity_thresh=float(cfg.get("proximity_thresh", 0.5)),
        appearance_thresh=float(cfg.get("appearance_thresh", 0.25)),
        # W trybie decoupled nie podpinamy feature-hooków YOLO → bezpieczniej bez lokalnego ReID
        with_reid=bool(cfg.get("with_reid", False)),
        model=str(cfg.get("model", "auto")),
        device=device,
        delta_t=int(cfg.get("delta_t", 3)),
        inertia=float(cfg.get("inertia", 0.2)),
        use_byte=bool(cfg.get("use_byte", True)),
        alpha_fixed_emb=float(cfg.get("alpha_fixed_emb", 0.95)),
        lost_match_thr=float(cfg.get("lost_match_thr", 0.0)),
        iou_weight=float(cfg.get("iou_weight", 0.5)),
        reid_weight=float(cfg.get("reid_weight", 0.5)),
        conf_weight=float(cfg.get("conf_weight", 0.1)),
        angle_weight=float(cfg.get("angle_weight", 0.05)),
        penalty_p=float(cfg.get("penalty_p", 0.2)),
        penalty_q=float(cfg.get("penalty_q", 0.4)),
        reduce_step=float(cfg.get("reduce_step", 0.05)),
        tai_thr=float(cfg.get("tai_thr", 0.55)),
        min_track_len=int(cfg.get("min_track_len", 3)),
    )


def _build_ultralytics_tracker(name: str, cfg: dict[str, Any], device=None):
    args = _ultralytics_args(cfg, device=device)
    if name == "ByteTrack":
        from ultralytics.trackers.byte_tracker import BYTETracker

        return BYTETracker(args)
    if name == "BoT-SORT":
        from ultralytics.trackers.bot_sort import BOTSORT

        return BOTSORT(args)
    if name == "Deep_OC-SORT":
        from ultralytics.trackers.deep_oc_sort import DeepOCSORT

        return DeepOCSORT(args)
    if name == "TrackTrack":
        from ultralytics.trackers.track_tracker import TRACKTRACK

        return TRACKTRACK(args)
    raise ValueError(f"Brak mapowania Ultralytics dla trackera: {name}")


def _build_local_tracker(name: str, cfg: dict[str, Any], video_name: str | None = None):
    if name == "Hybrid-SORT":
        from trackers.hybridsort import HybridSORTTracker

        return HybridSORTTracker(cfg)
    if name == "BoostTrack++":
        from trackers.boosttrackpp import BoostTrackPPTracker

        return BoostTrackPPTracker(cfg, video_name=video_name)
    raise ValueError(f"Brak lokalnej implementacji trackera: {name}")


class SCTTrackerAdapter:
    """
    Uniwersalna klasa opakowująca lokalny tracker SCT.

    - Wczytuje parametry z odpowiedniej sekcji experiment.yaml (przez get_sct_tracker_cfg).
    - Dla ByteTrack / BoT-SORT (oraz Deep_OC-SORT / TrackTrack) używa klas Ultralytics.
    - Dla Hybrid-SORT / BoostTrack++ używa `system/trackers/`.
    - `update(xyxy, conf, cls, frame)` → lista dict {box, track_id, score}.
    """

    def __init__(
        self,
        config: dict[str, Any],
        tracker_name: str | None = None,
        device=None,
        video_name: str | None = None,
    ):
        self.tracker_name = tracker_name or config.get("sct_tracker", {}).get("name", "BoT-SORT")
        self.source = resolve_tracker_source(self.tracker_name)
        if self.source is None:
            raise ValueError(
                f"Nieobsługiwany tracker SCT: '{self.tracker_name}'. "
                f"Dostępne: ByteTrack, BoT-SORT, Deep_OC-SORT, Hybrid-SORT, BoostTrack++, TrackTrack."
            )

        self.cfg = get_sct_tracker_cfg(config, self.tracker_name)
        self.device = device
        self.video_name = video_name

        if self.source == "ultralytics":
            # Lokalny ReID Ultralytics wymaga feature-hooków z model.track — w decoupled wyłączamy
            cfg_u = dict(self.cfg)
            cfg_u["with_reid"] = False
            self._backend = _build_ultralytics_tracker(self.tracker_name, cfg_u, device=device)
        else:
            self._backend = _build_local_tracker(self.tracker_name, self.cfg, video_name=video_name)

    def update(
        self,
        xyxy: np.ndarray,
        conf: np.ndarray,
        cls: np.ndarray,
        frame: np.ndarray,
    ) -> list[dict[str, Any]]:
        """
        Przekazuje detekcje z bieżącej klatki do trackera.

        Returns:
            Lista aktywnych torów: {"box": (x1,y1,x2,y2), "track_id": int, "score": float}
        """
        xyxy = np.asarray(xyxy, dtype=np.float32).reshape(-1, 4) if xyxy is not None else np.zeros((0, 4), np.float32)
        conf = np.asarray(conf, dtype=np.float32).reshape(-1) if conf is not None else np.zeros((0,), np.float32)
        cls = np.asarray(cls, dtype=np.float32).reshape(-1) if cls is not None else np.zeros((0,), np.float32)

        if self.source == "ultralytics":
            results = DetectionResults(xyxy, conf, cls)
            tracks = self._backend.update(results, frame)
            return self._parse_ultralytics_output(tracks)

        dets = np.concatenate([xyxy, conf.reshape(-1, 1)], axis=1) if len(xyxy) else np.empty((0, 5), np.float32)
        tracks = self._backend.update(dets, frame)
        return self._parse_local_output(tracks)

    @staticmethod
    def _parse_ultralytics_output(tracks) -> list[dict[str, Any]]:
        # [x1,y1,x2,y2,track_id,score,cls,idx]
        arr = np.asarray(tracks, dtype=np.float32)
        if arr.size == 0:
            return []
        if arr.ndim == 1:
            arr = arr.reshape(1, -1)
        out: list[dict[str, Any]] = []
        for row in arr:
            x1, y1, x2, y2 = map(float, row[:4])
            tid = int(row[4])
            score = float(row[5]) if row.shape[0] > 5 else 1.0
            out.append({"box": (x1, y1, x2, y2), "track_id": tid, "score": score})
        return out

    @staticmethod
    def _parse_local_output(tracks) -> list[dict[str, Any]]:
        # Hybrid: [x1,y1,x2,y2,id] ; BoostTrack++: [x1,y1,x2,y2,id,conf]
        arr = np.asarray(tracks, dtype=np.float32)
        if arr.size == 0:
            return []
        if arr.ndim == 1:
            arr = arr.reshape(1, -1)
        out: list[dict[str, Any]] = []
        for row in arr:
            x1, y1, x2, y2 = map(float, row[:4])
            tid = int(row[4])
            score = float(row[5]) if row.shape[0] > 5 else 1.0
            out.append({"box": (x1, y1, x2, y2), "track_id": tid, "score": score})
        return out
