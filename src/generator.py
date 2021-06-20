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
from augment import Aug, Hard_Aug, Transform_Aug, Flip_Aug



class Generator(keras.utils.Sequence):
    def __init__(
        self,
        df,
        batch_size,
        img_size,
        seed,
        prepared_img_path,
        n_classes,
        n_inputs=1,
        jpg=True,
        png=False,
        shuffle=False,
        augment=False,
        hard_augment=False,
        trans_aug=False,
        flip_aug=False,
        from_dicom=False,
    ):
        self.df = df.reset_index(drop=True)
        self.batch_size = batch_size
        self.img_size = img_size
        self.seed = seed
        self.prepared_img_path = prepared_img_path
        self.n_classes = n_classes
        self.n_inputs = n_inputs
        self.jpg = jpg
        self.png = png
        self._shuffle = shuffle
        self.augment = augment
        self.hard_augment = hard_augment
        self.trans_aug = trans_aug
        self.flip_aug = (flip_aug,)
        self.from_dicom = from_dicom

        if not os.path.exists(self.prepared_img_path):
            os.mkdir(self.prepared_img_path)

    def on_epoch_start(self):
        if self._shuffle:
            self.df = self.df.sample(frac=1, random_state=self.seed).reset_index(
                drop=True
            )

    def __len__(self):
        return self.df.shape[0] // self.batch_size

    def img_from_dicom(self, img_path, img_type):
        data_file = dicom.dcmread(img_path)
        img = data_file.pixel_array
        if img_type == "MONOCHROME1":
            img = img.max() - img
        img = (img - img.min()) / (img.max() - img.min())
        img = (np.array(img) * 255).astype("uint8")
        img = np.stack([img, img, img], axis=-1)
        img = tf.image.resize(
            img,
            (self.img_size, self.img_size),
        )
        img = tf.cast(img, tf.uint8)
        return img.numpy()

    def make_img(self, img_name, img_path, img_type):
        try:
            img = np.load(self.prepared_img_path + img_name + ".npy")
        except:
            if self.from_dicom:
                img = self.img_from_dicom(img_path, img_type)
            else:
                if self.jpg:
                    img = tf.io.read_file("/app/_data/jpg/" + img_name + ".jpg")
                    img = tf.image.decode_jpeg(img, channels=3)
                elif self.png:
                    img = tf.io.read_file("/app/_data/png/" + img_name + ".png")
                    img = tf.image.decode_png(img, channels=3)
                img = tf.image.resize(
                    img,
                    (self.img_size, self.img_size),
                )
                img = tf.cast(img, tf.uint8)
                img = img.numpy()
            np.save(self.prepared_img_path + img_name, img)
        return img

    def _get_one(self, ix):
        labels =  {'negative': 0, 'typical': 1, 'indeterminate': 2, 'atypical': 3}
        img_name = self.df.loc[ix, "image"][:-4]
        img_path = self.df.loc[ix, "path"]
        img_type = self.df.loc[ix, "PhotometricInterpretation"]
        img = self.make_img(img_name, img_path, img_type)
        label = self.df.loc[ix, "class"]
        with open("/app/_data/dict_metadata.json", "r") as f:
            dict_metadata = json.load(f)
        if self.augment:
            img = Aug.augment_image(img)
        if self.hard_augment:
            img = Hard_Aug.augment_image(img)
        if self.trans_aug:
            img = Transform_Aug.augment_image(img)
        if self.flip_aug:
            img = Flip_Aug.augment_image(img)
        x, y = {}, {}
        x["img"] = img
        if self.n_inputs == 2:
            modality = self.df.loc[ix, "modality"]
            PatientSex = self.df.loc[ix, "PatientSex"]
            body_part = self.df.loc[ix, "BodyPartExamined"]
            patient_sex_x = np.zeros(len(dict_metadata["PatientSex"]), dtype="uint8")
            body_part_x = np.zeros(
                len(dict_metadata["BodyPartExamined"]), dtype="uint8"
            )
            modality_x = np.zeros(len(dict_metadata["PatientSex"]), dtype="uint8")

            if PatientSex in dict_metadata["PatientSex"].keys():
                patient_sex_x[dict_metadata["PatientSex"][PatientSex]] = 1
            else:
                patient_sex_x[dict_metadata["PatientSex"]["unknown"]] = 1
            if body_part in dict_metadata["BodyPartExamined"].keys():
                body_part_x[dict_metadata["BodyPartExamined"][body_part]] = 1
            else:
                body_part_x[dict_metadata["BodyPartExamined"]["unknown"]] = 1
            if modality in dict_metadata["modality"].keys():
                modality_x[dict_metadata["modality"][modality]] = 1
            else:
                modality_x[dict_metadata["modality"]["unknown"]] = 1
            x["data"] = np.concatenate([patient_sex_x, body_part_x, modality_x])
        else:
            x["data"] = 0
        if self.n_classes == 4:
            y_ = np.zeros(4, dtype="uint8")
            y_[labels[label]] = 1
        elif self.n_classes == 1:
            if class_id == "negative":
                y_ = 1
            else:
                y_ = 0
        y["output"] = y_

        return x, y

    def __getitem__(self, batch_ix):
        x, y = {}, {}
        b_x_img, b_x_data, b_y = [], [], []
        for i in range(self.batch_size):
            x_dict, y_dict = self._get_one(i + self.batch_size * batch_ix)
            b_x_img.append(x_dict["img"])
            b_x_data.append(x_dict["data"])
            b_y.append(y_dict["output"])
        x["img"] = np.array(b_x_img)
        x["data"] = np.array(b_x_data)
        y["output"] = np.array(b_y)
        return x, y


class GetModel:
    def __init__(
        self,
        model_name,
        n_classes=4,
        n_inputs=2,
        lr=0.0005,
        top_dropout_rate=0.1,
        activation_func="softmax",
        weights=None,
        loss=keras.losses.CategoricalCrossentropy(from_logits=False),
        auc=keras.metrics.AUC(multi_label=True),
    ):
        self.model_name = model_name
        self.n_inputs = n_inputs
        self.lr = lr
        self.activation_func = activation_func
        self.weights = weights
        self.n_classes = n_classes
        self.top_dropout_rate = top_dropout_rate
        self.loss = loss
        self.auc = auc

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
        # img input
        input_img = keras.Input(shape=(IMG_SIZE, IMG_SIZE, 3), name="img")
        x_img = base_model(input_img)
        x_img = keras.layers.GlobalAveragePooling2D(name="avg_pool")(x_img)
        x_img = keras.layers.BatchNormalization()(x_img)
        if self.top_dropout_rate is not None:
            x_img = keras.layers.Dropout(self.top_dropout_rate, name="top_dropout")(
                x_img
            )
        # data input
        if self.n_inputs == 2:
            input_data = keras.Input(shape=18, name="data")
            x_data = keras.layers.Dense(32, activation="relu", name="dense_data_1")(
                input_data
            )
            x_data = keras.layers.Dense(32, activation="relu", name="dense_data_2")(
                x_data
            )
            x_data = keras.layers.Dense(32, activation="relu", name="dense_data_3")(
                x_data
            )
            x = keras.layers.Concatenate(axis=1, name="all")(
                [
                    x_img,
                    x_data,
                ]
            )
            inputs = [input_img, input_data]
        else:
            x = x_img
            inputs = input_img
        output = keras.layers.Dense(
            self.n_classes,
            activation=self.activation_func,
            dtype="float32",
            name="output",
        )(x)
        model = keras.Model(inputs=inputs, outputs=output)
        model.compile(
            loss=self.loss,
            optimizer=keras.optimizers.Adam(lr=self.lr),
            metrics=["acc", self.auc],
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



