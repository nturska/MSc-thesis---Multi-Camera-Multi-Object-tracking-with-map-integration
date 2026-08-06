import subprocess
import os

def run_evaluation():
    print("Inicjalizacja ewaluatora wizyjnego (TrackEval)...")
    
    # Główny skrypt modułu ewaluacyjnego
    trackeval_script = os.path.join("TrackEval", "scripts", "run_mot_challenge.py")
    
    # Wskazujemy katalog z pobraną prawdą referencyjną (Twoje pliki z Hugging Face)
    gt_folder = os.path.join("data", "MOT17", "train")
    
    # Wskazujemy katalog z Twoimi wynikami wygenerowanymi przez main.py
    trackers_folder = "results"
    
    # Wywołanie narzędzia TrackEval z precyzyjnymi argumentami
    command = [
        "python", trackeval_script,
        "--BENCHMARK", "MOT17",          # Typ ewaluacji
        "--SPLIT_TO_EVAL", "train",      # Ponieważ Ground Truth jest w folderze train
        "--TRACKERS_TO_EVAL", "MGR_Tracker", # Nazwa Twojego systemu
        "--GT_FOLDER", gt_folder,        # Ominięcie domyślnych ścieżek
        "--TRACKERS_FOLDER", trackers_folder,
        "--METRICS", "HOTA", "CLEAR", "Identity", # Kluczowe metryki z Twojej pracy!
        "--USE_PARALLEL", "False",
        "--NUM_PARALLEL_CORES", "1"
    ]
    
    try:
        print(f"Uruchamiam z folderu: {gt_folder}")
        subprocess.run(command, check=True)
        print("\nSukces! Zestawienie wyników wygenerowane.")
    except subprocess.CalledProcessError as e:
        print(f"\nBŁĄD: Ewaluacja zakończyła się niepowodzeniem. Kod: {e.returncode}")

if __name__ == '__main__':
    run_evaluation()