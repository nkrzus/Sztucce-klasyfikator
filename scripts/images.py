from PIL import Image
import os

# Ścieżka do folderu głównego z danymi
base_dir = '../data/train'  # Zmień na 'data/val' jeśli chcesz czyścić walidację

usuniete = 0

for klasa in os.listdir(base_dir):
    klasa_dir = os.path.join(base_dir, klasa)
    if not os.path.isdir(klasa_dir):
        continue
    for fname in os.listdir(klasa_dir):
        fpath = os.path.join(klasa_dir, fname)
        try:
            with Image.open(fpath) as img:
                img.verify()
        except Exception:
            print(f"❌ Usuwam uszkodzony plik: {fpath}")
            os.remove(fpath)
            usuniete += 1

print(f"\n✅ Gotowe! Usunięto {usuniete} plików.")
