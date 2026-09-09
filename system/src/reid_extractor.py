import os
import torch
import cv2
import numpy as np
from PIL import Image
from transformers import ViTImageProcessor, ViTModel

from src.device import get_torch_device

class ReIDExtractor:
    def __init__(self, model_name="google/vit-base-patch16-224-in21k", min_crop_size=10):
        self.device = get_torch_device()
        self.min_crop_size = min_crop_size
        print(f"Inicjalizacja modułu Re-ID (Vision Transformer) na: {self.device}")

        # 2. Ładowanie modelu z ekosystemu Hugging Face
        # Używamy bazowego modelu ViT. W docelowym wdrożeniu można go podmienić na
        # wagi modelu dostrojonego do Re-ID (np. TransReID trenowanego z Batch-Hard Triplet Loss).
        print(f"Pobieranie/Ładowanie modelu: {model_name} ...")

        self.processor = ViTImageProcessor.from_pretrained(model_name)
        self.model = ViTModel.from_pretrained(model_name).to(self.device)
        self.model.eval()  # Tryb wnioskowania (bez aktualizacji wag)

    def extract_features(self, frame, box):
        """
        Pobiera klatkę z kamery oraz współrzędne ramki (z YOLO) i zwraca wektor cech.
        """
        x1, y1, x2, y2 = map(int, box)

        # Zabezpieczenie przed wyjściem ramki poza kadr (np. gdy obiekt wchodzi w kadr)
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(frame.shape[1], x2), min(frame.shape[0], y2)

        # 1. Wycięcie sylwetki pieszego (kadr query)
        crop = frame[y1:y2, x1:x2]

        # Jeśli ramka jest pusta (błąd detekcji), zwracamy None
        if crop.size == 0 or crop.shape[0] < self.min_crop_size or crop.shape[1] < self.min_crop_size:
            return None
            
        # 2. Konwersja obrazu z formatu OpenCV (BGR) do formatu Hugging Face (RGB / PIL)
        crop_rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(crop_rgb)
        
        # 3. Wstępne przetwarzanie (zmiana rozmiaru do 224x224, normalizacja)
        inputs = self.processor(images=pil_img, return_tensors="pt").to(self.device)
        
        # 4. Predykcja (forward pass) bez obliczania gradientów
        with torch.no_grad():
            outputs = self.model(**inputs)
        
        # 5. Wyciągnięcie wektora cech
        # pooler_output zawiera zagregowaną reprezentację całego obrazu (1, 768)
        feature_vector = outputs.pooler_output.cpu().numpy()
        
        # Normalizacja wektora (L2), aby móc później używać podobieństwa cosinusowego
        feature_vector = feature_vector / np.linalg.norm(feature_vector)
        
        return feature_vector

# --- TESTOWANIE MODUŁU ---
if __name__ == '__main__':
    # Aby przetestować, użyjemy testowego obrazka lub losowej macierzy
    extractor = ReIDExtractor()
    
    # Tworzymy symulowaną klatkę wideo (np. 1080p, całkowicie czarna)
    dummy_frame = np.zeros((1080, 1920, 3), dtype=np.uint8)
    # Rysujemy na niej sztuczny biały prostokąt symulujący człowieka
    cv2.rectangle(dummy_frame, (100, 100), (300, 500), (255, 255, 255), -1)
    
    # Przekazujemy "sylwetkę" do modelu
    dummy_box = [100, 100, 300, 500]
    vector = extractor.extract_features(dummy_frame, dummy_box)
    
    if vector is not None:
        print(f"Sukces! Wygenerowano wektor cech o kształcie: {vector.shape}")
        print(f"Przykładowe wartości: {vector[0][:5]} ...")