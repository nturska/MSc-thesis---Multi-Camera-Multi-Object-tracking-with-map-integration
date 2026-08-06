import os
import torch
from ultralytics import YOLO, RTDETR

os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

def process_single_camera(video_path, use_rtdetr=False):
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    
    if use_rtdetr:
        model = RTDETR("models/rtdetr-l.pt") 
    else:
        model = YOLO("models/yolov8m.pt") 

    model.to(device)

    results = model.track(
        source=video_path,
        tracker="configs/custom_botsort.yaml",
        device=device,
        conf=0.2,          
        iou=0.5,                
        show=True,              
        stream=True,            
        classes=[0]             
    )

    for frame_idx, r in enumerate(results):
        if r.boxes.id is not None:
            track_ids = r.boxes.id.int().cpu().tolist()
            # print(f"Klatka {frame_idx}: ID = {track_ids}")

if __name__ == '__main__':
    # Uruchamiamy z modelem YOLOv8 Medium (z reguły szybszy i wystarczający przy dobrym trackerze)
    process_single_camera("data/videos/grzybowska.mp4", use_rtdetr=False)