import torch
import cv2
from ultralytics import YOLO
from transformers import AutoImageProcessor, AutoModel

def main():
    # Sprawdzanie architektury Apple Silicon (MPS)
    has_mps = torch.backends.mps.is_available()
    print(f"Czy akceleracja sprzętowa Apple (MPS) jest dostępna? : {has_mps}")
    
    if has_mps:
        # Przypisanie urządzenia do MPS
        device = torch.device("mps")
        print("Super! Twój kod będzie korzystał z procesora graficznego w Macu M5.")
    else:
        device = torch.device("cpu")
        print("UWAGA: MPS niedostępne, system będzie działał wolniej na procesorze (CPU).")

    # 1. Test YOLO
    print("\nŁadowanie modelu YOLO...")
    yolo_model = YOLO('yolov8n.pt')
    yolo_model.to(device) # Przeniesienie modelu na układ Apple
    print("YOLO załadowane pomyślnie!")

    # 2. Test Hugging Face
    print("\nŁadowanie modelu z Hugging Face...")
    reid_model_name = "google/vit-base-patch16-224-in21k"
    processor = AutoImageProcessor.from_pretrained(reid_model_name)
    reid_model = AutoModel.from_pretrained(reid_model_name)
    reid_model.to(device) # Przeniesienie modelu na układ Apple
    print("Model Hugging Face załadowany pomyślnie!")

    # 3. Test OpenCV
    print(f"\nWersja OpenCV: {cv2.__version__}")
    
    print("\nŚrodowisko gotowe do pracy na Macu M5! 🎉")

if __name__ == "__main__":
    main()