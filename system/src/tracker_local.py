import os
import sys
import torch
from ultralytics import YOLO, RTDETR

# Umożliwia uruchomienie jako python src/tracker_local.py z katalogu system/
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.config import load_config, write_botsort_yaml
from src.device import get_torch_device

os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"


def process_single_camera(video_path, use_rtdetr=False, config_path=None):
    config = load_config(config_path)
    detection_cfg = config["detection"]
    tracker_yaml = write_botsort_yaml(config["botsort"])

    device = get_torch_device()

    if use_rtdetr:
        model = RTDETR("models/rtdetr-l.pt")
    else:
        model_path = detection_cfg["model"]
        if not os.path.exists(model_path):
            candidate = f"models/{os.path.basename(model_path)}"
            if os.path.exists(candidate):
                model_path = candidate
        model = YOLO(model_path)

    model.to(device)

    results = model.track(
        source=video_path,
        tracker=tracker_yaml,
        device=device,
        conf=detection_cfg["conf"],
        iou=detection_cfg["iou"],
        show=True,
        stream=True,
        classes=detection_cfg["classes"],
    )

    for frame_idx, r in enumerate(results):
        if r.boxes.id is not None:
            track_ids = r.boxes.id.int().cpu().tolist()
            # print(f"Klatka {frame_idx}: ID = {track_ids}")


if __name__ == "__main__":
    # Domyślnie lokalny test SCT na sekwencji MOT17 (FRCNN)
    process_single_camera(
        "data/MOT17/train/MOT17-02-FRCNN/img1/%06d.jpg",
        use_rtdetr=False,
    )

