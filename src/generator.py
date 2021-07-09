# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.10.2
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

import json
import cv2
import os
import shutil
import albumentations
import numpy as np
import pandas as pd
import pydicom as dicom
import tensorflow as tf
from PIL import Image
from tensorflow import keras
import tensorflow_addons as tfa
from augment import Aug, Hard_Aug, Transform_Aug, Flip_Aug, Aug_No_transform

class Generator(keras.utils.Sequence):
    def __init__(
        self,
        df,
        label_columns,
        batch_size,
        img_size,
        seed,
        cache_img_path,
        shuffle=False,
        augment_fn=None,
        crop=False

    ):
        self.df = df.reset_index(drop=True)
        self.batch_size = batch_size
        self.img_size = img_size
        self.seed = seed
        self.cache_img_path = cache_img_path
        self._shuffle = shuffle
        self.augment_fn = augment_fn
        self.label_columns = label_columns
        self.crop = crop

        if not os.path.exists(self.cache_img_path):
            os.mkdir(self.cache_img_path)

    def on_epoch_start(self):
        if self._shuffle:
            self.df = self.df.sample(frac=1, random_state=self.seed).reset_index(
                drop=True
            )
    
    def __len__(self):
        return self.df.shape[0] // self.batch_size

    def img_from_dicom(self, img_path, img_type, coord):
        data_file = dicom.dcmread(img_path)
        img = data_file.pixel_array
        if img_type == "MONOCHROME1":
            img = img.max() - img
        img = (img - img.min()) / (img.max() - img.min())
        img = (np.array(img) * 255).astype("uint8")
        if self.crop:
            xmin = coord[0]*img.shape[1]
            ymin = coord[1]*img.shape[0]
            xmax = coord[2]*img.shape[1]
            ymax = coord[3]*img.shape[0]
            img = img[int(ymin) : int(ymax), int(xmin) : int(xmax)]
            
        img = np.stack([img, img, img], axis=-1)
        img = tf.image.resize(
            img,
            (self.img_size, self.img_size),
        )
        img = tf.cast(img, tf.uint8)
        return img.numpy()

    def make_img(self, img_name, img_path, img_type, coord):
        try:
            img = np.load(self.cache_img_path + img_name + ".npy")
        except:
            img = self.img_from_dicom(img_path, img_type, coord)
            np.save(self.cache_img_path + img_name, img)
        return img

    def _get_one(self, ix):
        img_name = self.df.loc[ix, "image"]
        img_path = self.df.loc[ix, "path"]
        img_type = self.df.loc[ix, "PhotometricInterpretation"]
        if self.crop:
            coord = self.df.loc[ix, ['xmin', 'ymin', 'xmax', 'ymax']].values
        else:
            coord=None
        img = self.make_img(img_name, img_path, img_type, coord)
        if self.label_columns is not None:
            if len(self.label_columns)==1:
                labels = self.df.loc[ix, self.label_columns].astype('uint8')
            else:
                labels = self.df.loc[ix, self.label_columns].astype('uint8').values
        if self.augment_fn is not None:
            try:
                img = self.augment_fn(img)
            except:
                img = img
        x = img
        y = labels if self.label_columns is not None else 0

        return x, y

    def __getitem__(self, batch_ix):
        x_, y_ = [],[]
        for i in range(self.batch_size):
            xx, yy = self._get_one(i + self.batch_size * batch_ix)
            x_.append(xx)
            y_.append(yy)
        x= np.array(x_)
        y = np.array(y_)
        return x, y

class GetModel:
    def __init__(
        self,
        model_name,
        n_classes=4,
        lr=0.0005,
        top_dropout_rate=0.1,
        activation_func="softmax",
        weights=None,
        loss=keras.losses.CategoricalCrossentropy(from_logits=False),
        metrics = ["acc", keras.metrics.AUC(multi_label=True)]
    ):
        self.model_name = model_name
        self.lr = lr
        self.activation_func = activation_func
        self.weights = weights
        self.n_classes = n_classes
        self.top_dropout_rate = top_dropout_rate
        self.loss = loss
        self.metrics = metrics
    def get_model(self):
        model_name = self.model_name
        with open("/app/_data/base_config.json", "r") as f:
            base_config = json.load(f)

        IMG_SIZE = base_config[model_name]["IMG_SIZE"]

        if model_name == "EFFB7":
            base_model = keras.applications.EfficientNetB7(
                weights=self.weights, include_top=False
            )
        elif model_name == "EFFB4":
            base_model = keras.applications.EfficientNetB4(
                weights=self.weights, include_top=False
            )
        elif model_name == "EFFB0":
            base_model = keras.applications.EfficientNetB0(
                weights=self.weights, include_top=False
            )
        elif model_name == "EFFB6":
            base_model = keras.applications.EfficientNetB6(
                weights=self.weights, include_top=False
            )
        # img input
        input_img = keras.Input(shape=(IMG_SIZE, IMG_SIZE, 3))
        x_img = base_model(input_img)
        x_img = keras.layers.GlobalAveragePooling2D(name="avg_pool")(x_img)
#         x_img = keras.layers.BatchNormalization()(x_img)
        if self.top_dropout_rate is not None:
            x_img = keras.layers.Dropout(self.top_dropout_rate, name="top_dropout")(
                x_img
            )
        x_img = keras.layers.Dense(
            128,
            activation='relu',
            dtype="float32"
        )(x_img)
        output = keras.layers.Dense(
            self.n_classes,
            activation=self.activation_func,
            dtype="float32"
        )(x_img)
        model = keras.Model(inputs=input_img, outputs=output)
        model.compile(
            loss=self.loss,
            optimizer=keras.optimizers.Adam(lr=self.lr),
            metrics=self.metrics,
        )
        return model

    def make_callback(
        self,
        model_path,
        model_name,
        tensorboard_path="/app/.tensorboard/",
        patience_ES=12,
        patience_RLR=5,
        factor_LR=0.9,
        metric_for_monitor="val_loss",
        metric_mode="min",
    ):
        if not os.path.exists(model_path):
            os.makedirs(model_path)
        else:
            print("Warning! Model path already exists.")
        callbacks = [
            keras.callbacks.EarlyStopping(
                monitor=metric_for_monitor,
                patience=patience_ES,
                restore_best_weights=True,
                verbose=1,
                mode=metric_mode,
            ),
            keras.callbacks.ModelCheckpoint(
                filepath=model_path + model_name,
                monitor=metric_for_monitor,
                verbose=1,
                save_best_only=True,
                save_weights_only=False,
                mode=metric_mode,
                save_freq="epoch",
            ),
            keras.callbacks.ReduceLROnPlateau(
                monitor=metric_for_monitor,
                factor=factor_LR,
                patience=patience_RLR,
                verbose=1,
                mode=metric_mode,
                min_delta=1e-5,
                min_lr=0.00000001,
            ),
            keras.callbacks.TensorBoard(log_dir=tensorboard_path, histogram_freq=0),
            keras.callbacks.experimental.BackupAndRestore(model_path + "backup/"),
            keras.callbacks.TerminateOnNaN(),
        ]
        return callbacks