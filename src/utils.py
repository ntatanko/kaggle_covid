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

class Aug:
    def augment_image(img):
        h = img.shape[0]
        w = img.shape[1]
        transform = albumentations.Compose(
            [
                albumentations.CLAHE(p=0.1, clip_limit=(1, 2), tile_grid_size=(8, 8)),
                albumentations.OneOf(
                    [
                        albumentations.MotionBlur((3, 5)),
                        albumentations.MedianBlur(blur_limit=5),
                        albumentations.GaussianBlur(blur_limit=(3, 5), sigma_limit=0),
                        albumentations.Blur(blur_limit=(3, 5)),
                    ],
                    p=0.2,
                ),
                albumentations.OneOf(
                    [
                        albumentations.GaussNoise(var_limit=[10, 20], mean=1),
                        albumentations.ImageCompression(
                            quality_lower=70, quality_upper=100, compression_type=1
                        ),
                        albumentations.MultiplicativeNoise(
                            multiplier=(0.95, 1.05), per_channel=False, elementwise=True
                        ),
                        albumentations.Downscale(
                            scale_min=0.8, scale_max=0.99, interpolation=4
                        ),
                    ],
                    p=0.2,
                ),
                albumentations.OneOf(
                    [
                        albumentations.RandomBrightnessContrast(
                            brightness_limit=0.2,
                            contrast_limit=0.2,
                            brightness_by_max=True,
                        ),
                        albumentations.augmentations.transforms.Sharpen(
                            alpha=(0.05, 0.15), lightness=(0.5, 1.0), p=1
                        ),
                        albumentations.augmentations.transforms.RandomToneCurve(
                            scale=0.15
                        ),
                    ],
                    p=0.2,
                ),
                albumentations.RandomSizedCrop(
                    min_max_height=(0.9 * h, 0.9 * w),
                    height=h,
                    width=w,
                    w2h_ratio=1.0,
                    interpolation=0,
                    p=0.2,
                ),
                albumentations.OneOf(
                    [
                        albumentations.OpticalDistortion(
                            distort_limit=0.1,
                            shift_limit=0.1,
                            border_mode=0,
                        ),
                        albumentations.ElasticTransform(
                            alpha=2.0,
                            sigma=2.0,
                            alpha_affine=2.0,
                            interpolation=0,
                            border_mode=0,
                        ),
                        albumentations.GridDistortion(
                            num_steps=5,
                            distort_limit=0.3,
                            interpolation=0,
                            border_mode=0,
                        ),
                    ],
                    p=0.3,
                ),
                albumentations.OneOf(
                    [
                        albumentations.Rotate(
                            limit=(-10, 10), interpolation=0, border_mode=0
                        ),
                        albumentations.ShiftScaleRotate(
                            shift_limit=0.1,
                            scale_limit=0.1,
                            rotate_limit=10,
                            interpolation=0,
                            border_mode=0,
                        ),
                        albumentations.augmentations.crops.transforms.CropAndPad(
                            px=None,
                            percent=(-0.15, 0.1),
                            pad_mode=0,
                            pad_cval=0,
                            pad_cval_mask=0,
                            keep_size=True,
                            sample_independently=True,
                            interpolation=1,
                        ),
                        albumentations.RandomSizedCrop(
                            min_max_height=(0.9 * h, 0.9 * w),
                            height=h,
                            width=w,
                            w2h_ratio=1.0,
                            interpolation=0,
                        ),
                    ],
                    p=0.3,
                ),
            ]
        )

        return transform(image=img)["image"]


class Generator(keras.utils.Sequence):
    def __init__(
        self,
        df,
        batch_size,
        img_size,
        seed,
        prepared_img_path,
        shuffle=False,
        augment=False,
    ):
        self.df = df.reset_index(drop=True)
        self.batch_size = batch_size
        self.img_size = img_size
        self.seed = seed
        self.prepared_img_path = prepared_img_path
        self._shuffle = shuffle
        self.augment = augment

        if not os.path.exists(self.prepared_img_path):
            os.mkdir(self.prepared_img_path)

    def on_epoch_start(self):
        if self._shuffle:
            self.df = self.df.sample(frac=1, random_state=self.seed).reset_index(
                drop=True
            )

    def __len__(self):
        return self.df.shape[0] // self.batch_size

    def make_img(self, img_name):
        try:
            img = np.load(self.prepared_img_path + img_name + ".npy")
        except:
            img = tf.io.read_file("/app/_data/jpg/" + img_name + ".jpg")
            img = tf.image.decode_jpeg(img, channels=3)
            img = tf.image.resize(
                img,
                (self.img_size, self.img_size),
            )
            img = tf.cast(img, tf.uint8)
            img = img.numpy()
            np.save(self.prepared_img_path + img_name, img)
        return img

    def _get_one(self, ix):
        img_name = self.df.loc[ix, "image"][:-4]
        class_id = self.df.loc[ix, "class"]
        modality = self.df.loc[ix, "modality"]
        PatientSex = self.df.loc[ix, "PatientSex"]
        body_part = self.df.loc[ix, "BodyPartExamined"]
        img = self.make_img(img_name)
        x = {}
        with open("/app/_data/dict_metadata.json", "r") as f:
            dict_metadata = json.load(f)
        if self.augment:
            img = Aug.augment_image(img)
        patient_sex_x = np.zeros(len(dict_metadata["PatientSex"]), dtype="uint8")
        body_part_x = np.zeros(len(dict_metadata["BodyPartExamined"]), dtype="uint8")
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
        x["img"] = img
        x["data"] = np.concatenate([patient_sex_x, body_part_x, modality_x])
        y = np.zeros(4, dtype="uint8")
        y[dict_metadata['class'][class_id]] = 1
        return x, y

    def __getitem__(self, batch_ix):

        x, y = {}, []
        b_x_img = []
        b_x_data = []
        for i in range(self.batch_size):
            x_dict, y_ = self._get_one(i + self.batch_size * batch_ix)
            b_x_img.append(x_dict["img"])
            b_x_data.append(x_dict["data"])
            y.append(y_)
        x["img"] = np.array(b_x_img)
        x["data"] = np.array(b_x_data)
        y = np.array(y)

        return x, y


class GetModel:
    def __init__(self, model_name):
        self.model_name = model_name

    def get_model(self):
        model_name  =self.model_name
        with open("/app/_data/base_config.json", "r") as f:
            base_config = json.load(f)

        IMG_SIZE = base_config[model_name]["IMG_SIZE"]
        weights_path = base_config[model_name]["WEIGHTS"]

        if model_name == "EFFB7":
            base_model = keras.applications.EfficientNetB7(
                weights=None, include_top=False
            )
        elif model_name == "EFFB4":
            base_model = keras.applications.EfficientNetB4(
                weights=None, include_top=False
            )
        base_model.load_weights(
            weights_path,
            by_name=True,
            skip_mismatch=True,
        )
        input_img = keras.Input(shape=(IMG_SIZE, IMG_SIZE, 3), name="img")
        x_img = base_model(input_img)
        x_img = keras.layers.GlobalAveragePooling2D(name="avg_pool")(x_img)

        input_data = keras.Input(shape=18, name="data")
        x_data = keras.layers.Dense(32, activation="relu", name="dense_data_1")(
            input_data
        )
        x_data = keras.layers.Dense(32, activation="relu", name="dense_data_2")(x_data)
        x_data = keras.layers.Dense(32, activation="relu", name="dense_data_3")(x_data)
        x = keras.layers.Concatenate(axis=1, name="all")(
            [
                x_img,
                x_data,
            ]
        )
        outputs = keras.layers.Dense(4, activation="sigmoid")(x)
        model = keras.Model(inputs=[input_img, input_data], outputs=outputs)
        model.compile(
            loss="binary_crossentropy",
            optimizer=keras.optimizers.Adam(lr=0.005),
            metrics=[
                "acc",
                keras.metrics.Recall(),
                keras.metrics.Precision(),
                tfa.metrics.F1Score(num_classes=4, average="weighted"),
            ],
        )
        return model

    def make_callback(self,
        model_path,
        model_name,
        tensorboard_path = '/app/.tensorboard/',
        patience_ES = 12,
        patience_RLR = 5,
        factor_LR = 0.9,
        metric_for_monitor='val_loss',
        metric_mode = 'min'
        
    ):
        if not os.path.exists(model_path):
            os.makedirs(model_path)
        else:
            print('Warning! Model path already exists.')
        callbacks = [
        keras.callbacks.EarlyStopping(
            monitor=metric_for_monitor,
            patience=patience_ES,
            restore_best_weights=True,
            verbose=1,
            mode=metric_mode,
        ),
        keras.callbacks.ModelCheckpoint(
            filepath = model_path+model_name,
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
        keras.callbacks.experimental.BackupAndRestore(model_path+"backup/"),
        keras.callbacks.TerminateOnNaN(),
    ]
        return callbacks