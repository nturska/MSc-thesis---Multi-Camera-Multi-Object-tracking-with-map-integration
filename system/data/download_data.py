import os
from dotenv import load_dotenv
from huggingface_hub import login, snapshot_download

# 1. Załadowanie tokenu z pliku .env
load_dotenv()
HF_TOKEN = os.getenv("HF_TOKEN")

if not HF_TOKEN:
    print("BŁĄD: Nie znaleziono tokenu w środowisku. Sprawdź plik .env!")
    exit()

# 2. GLOBALNA AUTORYZACJA
# Logujemy całą sesję Pythona, dzięki czemu żaden proces w tle nie wyśle 
# zapytania jako gość (usuwa to ostrzeżenie "unauthenticated requests").
print("Autoryzacja w Hugging Face Hub...")
login(token=HF_TOKEN)

# 3. POBIERANIE Z LIMITOWANIEM WĄTKÓW (Ochrona przed błędem 429)
print("\nPobieranie zbioru MOT17 (16 000+ plików - to potrwa kilka/kilkanaście minut)...")
snapshot_download(
    repo_id="Lekim89/MOT17", 
    repo_type="dataset", 
    local_dir="./data/MOT17",
    resume_download=True,
    max_workers=2  # Kluczowe: Pobieramy maksymalnie 2 pliki naraz!
)

print("\nPobieranie zbioru WILDTRACK...")
snapshot_download(
    repo_id="sein123/wildtrack", 
    repo_type="dataset", 
    local_dir="./data/WILDTRACK",
    resume_download=True,
    max_workers=2
)

print("\nZakończono pobieranie wszystkich zbiorów!")