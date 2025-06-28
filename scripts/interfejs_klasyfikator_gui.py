import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk
import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing import image
import os

# Załaduj model
model = tf.keras.models.load_model('../mobilenet_model.keras')
klasy = ['lyzka', 'noz', 'widelec']  # Kolejność zgodna z danymi treningowymi

# Funkcja predykcji
def przewidz_klase(img_path):
    img = image.load_img(img_path, target_size=(224, 224))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0) / 255.0
    pred = model.predict(x)[0]
    index = np.argmax(pred)
    return klasy[index], pred[index]

# Interfejs
plik_obrazka = None

def wczytaj_plik():
    global plik_obrazka
    plik_obrazka = filedialog.askopenfilename(filetypes=[("Obrazy", "*.jpg *.jpeg *.png")])
    if plik_obrazka:
        img = Image.open(plik_obrazka).resize((300, 300))
        img_tk = ImageTk.PhotoImage(img)
        canvas.image = img_tk
        canvas.create_image(0, 0, anchor='nw', image=img_tk)

        # Przewidz i zaktualizuj UI
        klasa, pewnosc = przewidz_klase(plik_obrazka)
        wynik_label.config(text=f"🧠 To wygląda na: {klasa.upper()} ({pewnosc*100:.1f}%)")
        popraw_label.config(text=f"Zgadzasz się z modelem?")
        btn_popraw.pack(pady=2)
        btn_ok.pack(pady=2)

def zapisz_etykiete(etykieta):
    if plik_obrazka:
        with open("etykiety_sprawdzone.txt", "a", encoding="utf-8") as f:
            f.write(f"{os.path.basename(plik_obrazka)}: {etykieta}\n")
        messagebox.showinfo("Zapisano!", f"Etykieta „{etykieta}” została zapisana.")
        wynik_label.config(text="")
        canvas.delete("all")
        btn_ok.pack_forget()
        btn_popraw.pack_forget()
        popraw_label.config(text="")

def popraw_etykiete():
    top = tk.Toplevel(root)
    top.title("Popraw etykietę")
    tk.Label(top, text="Wybierz właściwą klasę:").pack(pady=10)

    for klasa in klasy:
        tk.Button(top, text=klasa.upper(), width=15, command=lambda k=klasa: [zapisz_etykiete(k), top.destroy()]).pack(pady=5)

# GUI
root = tk.Tk()
root.title("Rozpoznawanie sztućców")

frame = tk.Frame(root)
frame.pack(padx=20, pady=20)

label = tk.Label(frame, text="Wczytaj zdjęcie i sprawdź co widzi model:", font=("Arial", 14))
label.pack()

canvas = tk.Canvas(frame, width=300, height=300)
canvas.pack(pady=10)

wynik_label = tk.Label(frame, text="", font=("Arial", 13, "bold"))
wynik_label.pack(pady=5)

popraw_label = tk.Label(frame, text="", font=("Arial", 11))
popraw_label.pack()

btn = tk.Button(frame, text="Wczytaj obraz", command=wczytaj_plik, font=("Arial", 12))
btn.pack(pady=10)

btn_ok = tk.Button(frame, text="✓ Zatwierdź etykietę", command=lambda: zapisz_etykiete(wynik_label.cget("text").split(':')[1].split('(')[0].strip().lower()))
btn_popraw = tk.Button(frame, text="✎ Zmień etykietę", command=popraw_etykiete)

root.mainloop()
