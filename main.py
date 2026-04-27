import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from PIL import Image, ImageEnhance
import os
import tkinter as tk
from tkinter import filedialog, messagebox
from tkinter import ttk
import matplotlib.pyplot as plt

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

def guimodelloader():
    win = tk.Tk()
    win.title("PseudoColor Project")
    win.geometry("450x180")
    win.configure(bg="#61053f")

    title = tk.Label(win, text="Load Colorization Model (.h5)",
                     font=("Times New Roman", 16, "bold"),
                     fg="white", bg="#61053f")
    title.pack(pady=10)

    model_path_var = tk.StringVar()

    entry = tk.Entry(win, textvariable=model_path_var, width=40, font=("Times New Roman", 11))
    entry.pack(pady=5)

    def browsingfile():
        path = filedialog.askopenfilename(
            title="Select Model",
            filetypes=[("H5 Model", "*.h5")]
        )
        model_path_var.set(path)

    ttk.Button(win, text="Browse", command=browsingfile).pack(pady=5)

    def loadingbutton():
        if not os.path.isfile(model_path_var.get()):
            messagebox.showerror("Error", "Invalid model file path.")
            return
        win.destroy()

    ttk.Button(win, text="Load Model", command=loadingbutton).pack(pady=10)

    win.mainloop()
    return model_path_var.get()


def postprocessing(img):
    rgb_uint8 = (np.clip(img, 0, 1) * 255).astype("uint8")
    lab = cv2.cvtColor(rgb_uint8, cv2.COLOR_RGB2LAB)
    L, A, B = cv2.split(lab)
    L = cv2.normalize(L, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)
    L = cv2.convertScaleAbs(L, alpha=1.12, beta=2)  
    L = np.clip(L, 0, 255).astype("uint8")
    enhanced_lab = cv2.merge([L, A, B])
    rgb_back = cv2.cvtColor(enhanced_lab, cv2.COLOR_LAB2RGB)
    pil_img = Image.fromarray(rgb_back.astype("uint8"))
    pil_img = ImageEnhance.Color(pil_img).enhance(1.35)
    pil_img = ImageEnhance.Contrast(pil_img).enhance(1.25)
    pil_img = ImageEnhance.Brightness(pil_img).enhance(1.10)
    pil_img = ImageEnhance.Sharpness(pil_img).enhance(1.15)

    return np.asarray(pil_img).astype("float32") / 255.0


def imagesinterface(model):
    win = tk.Tk()
    win.title("Image Colorization GUI")
    win.geometry("500x320")
    win.configure(bg="#61053f")

    lbl_title = tk.Label(win,
                         text="Grayscale → Colorized Output",
                         font=("Times New Roman", 18, "bold"),
                         fg="white", bg="#61053f")
    lbl_title.pack(pady=20)

    lbl_info = tk.Label(win,
                        text="Select an image to generate:\n"
                             "• Grayscale Input\n"
                             "• Colorized Output\n\n"
                             "You can choose NEW images repeatedly.",
                        font=("Times New Roman", 11),
                        fg="#cccccc", bg="#61053f")
    lbl_info.pack(pady=10)

    def imageprocessing():
        img_path = filedialog.askopenfilename(
            title="Select Image",
            filetypes=[("Images", "*.jpg *.jpeg *.png *.bmp")]
        )
        if not img_path:
            return

        bgr = cv2.imread(img_path)
        if bgr is None:
            messagebox.showerror("Error", "Could not read image.")
            return

        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
        g_resized = cv2.resize(gray, (256, 256))

        m_input = g_resized.astype("float32") / 255.0
        m_input = np.expand_dims(m_input, axis=[0, -1])

        print("Predicting...")
        output = model.predict(m_input)[0]
        output = cv2.resize(output, (rgb.shape[1], rgb.shape[0]))
        output = np.clip(output, 0, 1)

        colorized = postprocessing(output)


        plt.figure(figsize=(18, 8))

        plt.subplot(1, 2, 1)
        plt.imshow(gray, cmap="gray")
        plt.title("Grayscale Input", fontsize=16)
        plt.axis("off")

        plt.subplot(1, 2, 2)
        plt.imshow(colorized)
        plt.title("Colorized Output", fontsize=16)
        plt.axis("off")

        plt.show(block=False)

    ttk.Button(win, text="Select Image", command=imageprocessing).pack(pady=25)
    ttk.Button(win, text="Exit", command=win.destroy).pack(pady=10)

    win.mainloop()

if __name__ == "__main__":
    model_path = guimodelloader()

    print("Loading model...")
    model = load_model(model_path, compile=False)
    print("Model loaded successfully!")

    imagesinterface(model)
