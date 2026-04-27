----PseudoColor Image Colorization----

This project provides a simple and user-friendly GUI application that converts grayscale images into colorized images using a trained TensorFlow/Keras deep learning model.

You can:

Load any .h5 colorization model

Select images

View Grayscale Input and Colorized Output side-by-side

Re-select new images without restarting the program

View enhanced, clean, and natural colorized results

The application is built using:

Tkinter (Python GUI)

TensorFlow / Keras (Model inference)

OpenCV + Pillow (Image processing)

Matplotlib (Visualization)

----How It Works----

The app first asks you to choose a pre-trained colorization model (.h5)

After loading the model, a GUI window appears

You can select any grayscale image (JPG/PNG/BMP)

The model predicts the colorized output

Both images are displayed side-by-side:

Left: Grayscale Input

Right: Colorized Output

You can repeatedly select new images without closing the app

----Features----

✔ Model Loader GUI
✔ One-click Image Selection
✔ Color Enhancement (Brightness, Contrast, Sharpening, Color boost)
✔ Real-time Prediction using Model
✔ High-resolution output display
✔ Clean UI with professional fonts


----Installation----
1️⃣ Install Python

Make sure you have Python 3.8 – 3.12 installed.

2️⃣ Install required libraries

Run:

pip install tensorflow pillow opencv-python matplotlib

▶️ How to Run

Open terminal:

python main.py


A window will open asking you to Load Model (.h5)

Select your trained colorization model

Then GUI will appear → click Select Image

Choose a grayscale image → output will appear

Continue selecting new images as you like

Press Exit to close the app

----Supported Image Formats----

.jpg

.jpeg

.png

.bmp

----Output Enhancement----

The final output is improved with
Luminance Enhancement 

Improves lightness using LAB color space
Recovers details in shadows
Produces clearer, brighter colorized imagesh


Increased color saturation

Better overall contrast

Slight brightness boost

Mild sharpening

This produces a more visually appealing "Colorized Output".

----Model Input Requirements----

Your TensorFlow/Keras model must:

Accept grayscale images of size 256×256×1

Output a color image (RGB) of the same size (after resizing)

