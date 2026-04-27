import cv2
import numpy as np

def rimage(path):
    return cv2.imread(path)
def conversionrgb(image_bgr):
    return cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
def resizeimage(image_rgb, target_size):
    return cv2.resize(image_rgb, target_size)
def n_image(image_resized):
    return image_resized.astype(np.float32) / 255.0
def l_image(path, target_size=(256, 256)):
    image_bgr = rimage(path)
    image_rgb = conversionrgb(image_bgr)
    image_res = resizeimage(image_rgb, target_size)
    image_n = n_image(image_res)
    return image_n
