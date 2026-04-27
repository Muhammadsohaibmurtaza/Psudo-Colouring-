
import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D, Concatenate
from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, EarlyStopping, Callback
import matplotlib.pyplot as plt

Sizeofimage = 256
Sizeofbatch = 4
numberofepochs = 30
pathofthedataset = "trainimages"
finalmodelname = "color_unet_custom.h5"   ## best model

def returnlistofimages(f_path):
    i_file = []
    for f in os.listdir(f_path):
        if f.lower().endswith((".jpg", ".jpeg", ".png")):
            i_file.append(os.path.join(f_path, f))
    if not i_file:
        raise ValueError("No images found in dataset folder!")
    return np.array(i_file)

def dataset_splitting(f_list, t_ratio=0.8):
    np.random.shuffle(f_list)
    split_index = int(len(f_list) * t_ratio)
    return f_list[:split_index], f_list[split_index:]

def preprocessingimages(path):
    ibytes = tf.io.read_file(path)
    irgb = tf.image.decode_jpeg(ibytes, channels=3)
    iresized = tf.image.resize(irgb, (Sizeofimage, Sizeofimage))
    inormalized = tf.cast(iresized, tf.float32) / 255.0
    igray = tf.image.rgb_to_grayscale(inormalized)
    return igray, inormalized

def datasetcreating(f_paths, batchsizes, shuffle=False):
    ds = tf.data.Dataset.from_tensor_slices(f_paths)
    ds = ds.map(preprocessingimages, num_parallel_calls=tf.data.AUTOTUNE)
    if shuffle:
        ds = ds.shuffle(buffer_size=200)
    ds = ds.batch(batchsizes).prefetch(tf.data.AUTOTUNE)
    return ds


def convulationallayers(x, filters):
    x = Conv2D(filters, 3, activation="relu", padding="same")(x)
    x = Conv2D(filters, 3, activation="relu", padding="same")(x)
    return x

def buildmodel(input_shape=(Sizeofimage, Sizeofimage, 1)):
    i = Input(input_shape)
    
    c1 = convulationallayers(i, 64); p1 = MaxPooling2D(2)(c1)
    c2 = convulationallayers(p1, 128); p2 = MaxPooling2D(2)(c2)
    c3 = convulationallayers(p2, 256); p3 = MaxPooling2D(2)(c3)
    c4 = convulationallayers(p3, 512); p4 = MaxPooling2D(2)(c4)

    bn = convulationallayers(p4, 1024)

    u1 = UpSampling2D(2)(bn); u1 = Concatenate()([u1, c4]); c5 = convulationallayers(u1, 512)
    u2 = UpSampling2D(2)(c5); u2 = Concatenate()([u2, c3]); c6 = convulationallayers(u2, 256)
    u3 = UpSampling2D(2)(c6); u3 = Concatenate()([u3, c2]); c7 = convulationallayers(u3, 128)
    u4 = UpSampling2D(2)(c7); u4 = Concatenate()([u4, c1]); c8 = convulationallayers(u4, 64)

    outputs = Conv2D(3, 1, activation="sigmoid")(c8)
    return tf.keras.Model(i, outputs)

class colorpreviewclass(Callback):
    def __init__(s, ds_value):
        super().__init__()
        s.ds_value = ds_value  

    def on_epoch_end(s, epoch, logs=None):   
        g_batch, c_batch = next(iter(s.ds_value))
        pred_batch = s.model.predict(g_batch)  

        plt.figure(figsize=(12, 4))
        plt.subplot(1, 3, 1)
        plt.imshow(tf.squeeze(g_batch[0]), cmap="gray")
        plt.title("Gray Input"); plt.axis("off")

        plt.subplot(1, 3, 2)
        plt.imshow(c_batch[0])
        plt.title("Original"); plt.axis("off")

        plt.subplot(1, 3, 3)
        plt.imshow(pred_batch[0])
        plt.title("Predicted Color"); plt.axis("off")

        plt.show()


def training():
    listofallfiles = returnlistofimages(pathofthedataset)
    t_files, v_files = dataset_splitting(listofallfiles)

    print(f"Total images: {len(listofallfiles)}")
    print(f"Train: {len(t_files)}")
    print(f"Validation: {len(v_files)}")

    t_ds = datasetcreating(t_files, Sizeofbatch, shuffle=True)
    v_ds = datasetcreating(v_files, Sizeofbatch)

    model = buildmodel()
    model.compile(optimizer=tf.keras.optimizers.Adam(1e-4), loss="mse")
    print(model.summary())

    savingcheckpoints = ModelCheckpoint(finalmodelname, save_best_only=True, monitor="val_loss", mode="min", verbose=1)
    lr_cb = ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=3)
    early_cb = EarlyStopping(monitor="val_loss", patience=6, restore_best_weights=True)
    preview_cb = colorpreviewclass(v_ds)

    history = model.fit(
        t_ds,
        validation_data=v_ds,
        epochs=numberofepochs,  
        callbacks=[savingcheckpoints, lr_cb, early_cb, preview_cb]
    )

    print("Training completed")
    return history, model

if __name__ == "__main__":
    training()
