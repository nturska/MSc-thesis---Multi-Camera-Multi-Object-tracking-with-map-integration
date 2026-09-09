import cv2
import numpy as np
import torch
import os
import sys
from ultralytics import YOLO

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.device import get_torch_device

def main():
    # Ścieżki
    video_path = "../system/data/videos/grzybowska.mp4"
    
    # 1. Inicjalizacja sprzętu i modelu YOLO (z BoT-SORT)
    device = get_torch_device()
    model = YOLO("../system/models/yolov8m.pt")
    model.to(device)

    # 2. Inicjalizacja algorytmu wyodrębniania tła (MOG2)
    backSub = cv2.createBackgroundSubtractorMOG2(history=500, varThreshold=16, detectShadows=True)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("Błąd wczytywania wideo.")
        return

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # A) GENEROWANIE OBRAZU 'B' (Rzeczywisty ruch wg MOG2)
        mask_B = backSub.apply(frame)
        _, mask_B = cv2.threshold(mask_B, 254, 255, cv2.THRESH_BINARY)
        mask_B = cv2.morphologyEx(mask_B, cv2.MORPH_OPEN, kernel) # Czysta maska tła

        # B) GENEROWANIE OBRAZU 'A' (Syntetyczny obraz na podstawie YOLO)
        # Tworzymy całkowicie czarny obraz o wymiarach naszej klatki wideo
        height, width = frame.shape[:2]
        synthetic_A = np.zeros((height, width), dtype=np.uint8)

        # Uruchamiamy śledzenie
        results = model.track(frame, tracker="../system/configs/custom_botsort.yaml", device=device, persist=True, verbose=False, classes=[0], conf=0.2)
        
        if results[0].boxes.id is not None:
            boxes = results[0].boxes.xyxy.cpu().numpy()
            
            for box in boxes:
                x1, y1, x2, y2 = map(int, box)
                # Rysujemy pełny, biały prostokąt w miejscu, gdzie YOLO widzi człowieka
                cv2.rectangle(synthetic_A, (x1, y1), (x2, y2), 255, -1)
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

        # C) OBLICZANIE KARY - Różnica symetryczna (XOR)
        # To bezpośrednia implementacja wzoru: B ⊗ (1 - A) + (1 - B) ⊗ A
        # Białe piksele na tym obrazie to błędy (brak zgodności między AI a ruchem fizycznym)
        penalty_mask = cv2.bitwise_xor(mask_B, synthetic_A)

        # Skalowanie w dół dla łatwiejszego wyświetlania 4 okien na ekranie
        scale = 0.5
        frame_resized = cv2.resize(frame, (0,0), fx=scale, fy=scale)
        mask_B_resized = cv2.resize(mask_B, (0,0), fx=scale, fy=scale)
        synthetic_A_resized = cv2.resize(synthetic_A, (0,0), fx=scale, fy=scale)
        penalty_resized = cv2.resize(penalty_mask, (0,0), fx=scale, fy=scale)

        # Wyświetlanie
        cv2.imshow("1. Oryginalne detekcje (YOLO)", frame_resized)
        cv2.imshow("2. Obraz B: Rzeczywisty ruch (MOG2)", mask_B_resized)
        cv2.imshow("3. Obraz A: Syntetyczne rzuty", synthetic_A_resized)
        cv2.imshow("4. Pseudoodleglosc POM (Blad rzutowania)", penalty_resized)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()