"""BoostTrack++ — oficjalna implementacja (vukasin-stanojevic/BoostTrack), bez zewnętrznego ReID."""

from __future__ import annotations

from typing import Any

import numpy as np

from . import settings as bt_settings
from .boost_track import BoostTrack


def _apply_config(cfg: dict[str, Any]) -> None:
    """Nadpisuje globalne ustawienia BoostTrack wartościami z experiment.yaml."""
    g = bt_settings.GeneralSettings.values
    if "det_thresh" in cfg or "track_high_thresh" in cfg:
        g["det_thresh"] = float(cfg.get("det_thresh", cfg.get("track_high_thresh", g["det_thresh"])))
    if "iou_threshold" in cfg or "match_thresh" in cfg:
        # paper iou_threshold ≈ min IoU; u nas trzymamy jako iou_threshold BoostTrack
        g["iou_threshold"] = float(cfg.get("iou_threshold", 0.3))
    if "track_buffer" in cfg or "max_age" in cfg:
        g["max_age"] = int(cfg.get("track_buffer", cfg.get("max_age", g["max_age"])))
    if "min_hits" in cfg:
        g["min_hits"] = int(cfg["min_hits"])
    g["use_embedding"] = bool(cfg.get("use_embedding", False))
    g["use_ecc"] = bool(cfg.get("use_ecc", True))

    btpp = bt_settings.BoostTrackPlusPlusSettings.values
    # Mapowanie beta_high/beta_low/varying_frames → soft/varying threshold (BoostTrack++)
    if "beta_high" in cfg:
        btpp["use_vt"] = True
    if "beta_low" in cfg:
        btpp["use_sb"] = True
    btpp["use_rich_s"] = bool(cfg.get("use_rich_s", True))
    btpp["use_sb"] = bool(cfg.get("use_sb", True))
    btpp["use_vt"] = bool(cfg.get("use_vt", True))

    # Zachowaj progi varying-threshold w GeneralSettings (używane w wrapperze override)
    g["_beta_high"] = float(cfg.get("beta_high", 0.95))
    g["_beta_low"] = float(cfg.get("beta_low", 0.8))
    g["_varying_frames"] = int(cfg.get("varying_frames", 20))


class BoostTrackPPTracker:
    """
    Wrapper wokół oficjalnego BoostTrack (tryb BoostTrack++).
    Wejście: dets (N,5) = [x1,y1,x2,y2,score] w współrzędnych obrazu.
    Wyjście: (M,6) = [x1,y1,x2,y2,track_id,confidence]
    """

    def __init__(self, cfg: dict[str, Any] | None = None, video_name: str | None = None):
        cfg = dict(cfg or {})
        _apply_config(cfg)
        self._beta_high = float(bt_settings.GeneralSettings.values.get("_beta_high", 0.95))
        self._beta_low = float(bt_settings.GeneralSettings.values.get("_beta_low", 0.8))
        self._varying_frames = int(bt_settings.GeneralSettings.values.get("_varying_frames", 20))
        self.tracker = BoostTrack(video_name=video_name)
        # Podmień hardcoded varying-threshold, jeśli BoostTrack++ VT jest włączone
        self._patch_varying_threshold()

    def _patch_varying_threshold(self) -> None:
        orig = self.tracker.dlo_confidence_boost

        def patched(detections, use_rich_sim, use_soft_boost, use_varying_th):
            # logika VT z oficjalnego BoostTrack++ + progi z experiment.yaml
            sbiou_matrix = self.tracker.get_iou_matrix(detections, True)
            if sbiou_matrix.size == 0:
                return detections

            trackers = np.zeros((len(self.tracker.trackers), 6))
            for t, trk in enumerate(trackers):
                pos = self.tracker.trackers[t].get_state()[0]
                trk[:] = [pos[0], pos[1], pos[2], pos[3], 0, self.tracker.trackers[t].time_since_update - 1]

            if use_rich_sim:
                from .assoc import MhDist_similarity, shape_similarity

                mhd_sim = MhDist_similarity(self.tracker.get_mh_dist_matrix(detections), 1)
                shape_sim = shape_similarity(detections, trackers)
                S = (mhd_sim + shape_sim + sbiou_matrix) / 3
            else:
                S = self.tracker.get_iou_matrix(detections, False)

            if not use_soft_boost and not use_varying_th:
                max_s = S.max(1)
                coef = self.tracker.dlo_boost_coef
                detections[:, 4] = np.maximum(detections[:, 4], max_s * coef)
            else:
                if use_soft_boost:
                    max_s = S.max(1)
                    alpha = 0.65
                    detections[:, 4] = np.maximum(
                        detections[:, 4], alpha * detections[:, 4] + (1 - alpha) * max_s ** (1.5)
                    )
                if use_varying_th:
                    threshold_s = self._beta_high
                    threshold_e = self._beta_low
                    n_steps = max(self._varying_frames, 1)
                    alpha = (threshold_s - threshold_e) / n_steps
                    tmp = (S > np.maximum(threshold_s - trackers[:, 5] * alpha, threshold_e)).max(1)
                    scores = detections[:, 4].copy()
                    scores[tmp] = np.maximum(scores[tmp], self.tracker.det_thresh + 1e-5)
                    detections[:, 4] = scores
            return detections

        self.tracker.dlo_confidence_boost = patched  # type: ignore[method-assign]

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

        class _FakeTensor:
            # BoostTrack liczy scale = min(H_tensor/H_img, W_tensor/W_img); chcemy scale=1
            shape = (1, 3, h, w)

        out = self.tracker.update(dets, _FakeTensor(), frame, tag="live")
        if out is None or len(out) == 0:
            return np.empty((0, 5), dtype=np.float32)
        return np.asarray(out, dtype=np.float32)
