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

# +
import json
import os
import shutil
import re
import numpy as np
import pandas as pd
import pydicom as dicom
import tensorflow as tf
import tensorflow_addons as tfa
import torch
from PIL import Image
from tensorflow import keras
from tqdm import tqdm
import matplotlib.pyplot as plt
from numba import cuda

# -

class Test:
    def prepare_data(data_path, path_to_save, train_mode=True, n_train_sample=10, jpg=False):
        if not os.path.exists(path_to_save):
            os.mkdir(path_to_save)
        test_df = pd.DataFrame()
        if train_mode:
            p = os.listdir(data_path)[:n_train_sample]
        else:
            p = os.listdir(data_path)
        for folder1 in tqdm(p):
            for folder2 in os.listdir(data_path + folder1):
                path = os.path.join(data_path, folder1, folder2)
                for img_file in os.listdir(path):
                    img_path = os.path.join(path, img_file)
                    test_df.loc[img_file, "path"] = img_path
                    data_file = dicom.dcmread(img_path)
                    test_df.loc[img_file, "StudyInstanceUID"] = folder1
                    test_df.loc[img_file, "id_image"] = img_file[:-4]
                    test_df.loc[img_file, "modality"] = data_file.Modality
                    test_df.loc[img_file, "PatientSex"] = data_file.PatientSex
                    test_df.loc[
                        img_file, "BodyPartExamined"
                    ] = data_file.BodyPartExamined
                    test_df.loc[
                        img_file, "PhotometricInterpretation"
                    ] = data_file.PhotometricInterpretation
                    test_df.loc[img_file, "width"] = data_file.pixel_array.shape[1]
                    test_df.loc[img_file, "height"] = data_file.pixel_array.shape[0]
                    img = data_file.pixel_array
                    if data_file.PhotometricInterpretation == "MONOCHROME1":
                        img = img.max() - img
                    img = (img - img.min()) / (img.max() - img.min())
                    img = (np.array(img) * 255).astype("uint8")
                    img = np.stack([img, img, img], axis=-1)
                    img = Image.fromarray(img)
                    if jpg:
                        img.save(path_to_save + img_file[:-4] + ".jpg", format='JPEG')
                    else:
                        img.save(path_to_save + img_file[:-4] + ".png", format='png')
        test_df = test_df.reset_index(drop=True)
        test_df["path"] = test_df["path"].str.replace("\../", "/kaggle/")
        return test_df

    def make_classification(
        test_df_,
        jpg_path,
        metadata_path,
        model_path,
        model_2cl_path,
        labels,
        obj_det=True,
        classification=True,
        img_size=600,
        img_from_folder=True,
        n_inputs=2,
    ):
        sub_study = pd.DataFrame(columns=["id", "PredictionString"])
        gen = Generator(
            df=test_df_,
            img_size=img_size,
            jpg_path=jpg_path,
            metadata_path=metadata_path,
            img_from_folder=img_from_folder,
            batch_size=1,
            n_inputs=n_inputs,
        )
        print(gen.__getitem__(2)["img"].shape)
        plt.imshow(gen.__getitem__(1)["img"][0])
        plt.xticks([])
        plt.yticks([])
        plt.show()

        # check only object detection score
        if not classification:
            for ix in test_df_.StudyInstanceUID.unique().tolist():
                sub_study = sub_study.append(
                    {
                        "id": ix + "_study",
                        "PredictionString": "negative 1 0 0 1 1 typical 1 0 0 1 1 indeterminate 1 0 0 1 1 atypical 1 0 0 1 1",
                    },
                    ignore_index=True,
                )
        # classification 4 classes: "negative", "typical", "indeterminate", "atypical"
        else:
            eff_models = []
            for m_path in model_path:
                if ".h5" in m_path:
                    eff_models.append(keras.models.load_model(m_path))
                else:
                    for file in os.listdir(m_path):
                        if ".h5" in file:
                            eff_models.append(keras.models.load_model(m_path + file))
                        # classification 2 classes: "negative", "positive"
            if model_2cl_path is not None:
                models_2cl = []
                for m_path in model_2cl_path:
                    models_2cl.append(keras.models.load_model(m_path))

            for ix in tqdm(test_df_.index.tolist()):
                data = gen.__getitem__(ix)
                pred = np.mean(
                    [model(data)[0] for model in eff_models],
                    axis=0
                )
                pred_neg = np.mean(
                    [m(data)[0][0] for m in models_2cl]
                    )
                test_df_.loc[ix, "negative"] = pred[0]
                test_df_.loc[ix, "typical"] = pred[1]
                test_df_.loc[ix, "indeterminate"] = pred[2]
                test_df_.loc[ix, "atypical"] = pred[3]
                test_df_.loc[ix, "negative_2cl"] = pred_neg


            # PredictionString in format 'label threshold 0 0 1 1'
            groupped_df = test_df_.groupby("StudyInstanceUID")[
                ["negative", "typical", "indeterminate", "atypical"]
            ].mean()
            for ix in groupped_df.index.tolist():
                predictions = list(map(str, (np.round(groupped_df.loc[ix].values, 5))))
                pred = " ".join(
                    [
                        labels[i] + " " + predictions[i] + " 0 0 1 1"
                        for i in range(len(labels))
                    ]
                )
                sub_study = sub_study.append(
                    {"id": ix + "_study", "PredictionString": pred}, ignore_index=True
                )

        # check only classification score
        if not obj_det:
            sub_image = pd.DataFrame(columns=["id", "PredictionString"])
            for img_name in test_df_.id_image.tolist():
                sub_image = sub_image.append(
                    {"id": img_name + "_image", "PredictionString": "none 1 0 0 1 1"},
                    ignore_index=True,
                )
            return sub_study, sub_image, test_df_
        return sub_study, test_df_

    def make_bbox_df(
        test_df_,
        SAVE_BBOX_PATH,
        bigger_better = True,
        neg_4cl = False,
        conf_1 = False,
    ):
        sub_image = pd.DataFrame(columns=["id", "PredictionString"])
        if os.path.exists(SAVE_BBOX_PATH + "labels/"):
            SAVE_BBOX_PATH = SAVE_BBOX_PATH + "labels/"
        else:
            SAVE_BBOX_PATH = SAVE_BBOX_PATH
        for file in os.listdir(SAVE_BBOX_PATH):
            if ".txt" in file:
                img_name = file[:-4]
                w = test_df_[test_df_["id_image"] == img_name]["width"].values[0]
                h = test_df_[test_df_["id_image"] == img_name]["height"].values[0]
                with open(SAVE_BBOX_PATH + file, "r") as f:
                    ls = f.read()
                    ls = re.sub(r"[\n]", " ", ls).strip().split()
                    ls = list(map(float, ls))
                    list_pred = []
                    for i in range(0, len(ls), 6):
                        x_c = ls[i + 1]
                        y_c = ls[i + 2]
                        w_p = ls[i + 3]
                        h_p = ls[i + 4]
                        conf = ls[i + 5]
                        xmin = int((x_c - w_p / 2) * w)
                        ymin = int((y_c - h_p / 2) * h)
                        xmax = int((x_c + w_p / 2) * w)
                        ymax = int((y_c + h_p / 2) * h)
                        list_pred.extend(["opacity", conf, xmin, ymin, xmax, ymax])
                    sub_image = sub_image.append(
                        {
                            "id": img_name + "_image",
                            "PredictionString": " ".join(list(map(str, list_pred))),
                        },
                        ignore_index=True,
                    )
        for img_name in test_df_.id_image.tolist():
            if img_name + "_image" not in sub_image.id.tolist():
                if conf_1:
                    sub_image = sub_image.append(
                    {
                        "id": img_name + "_image",
                        "PredictionString": "none 1 0 0 1 1",
                    },
                    ignore_index=True,
                )
                else:
                    negative_2cl = test_df_[test_df_["id_image"] == img_name]['negative_2cl'].values[0]
                    negative = test_df_[test_df_["id_image"] == img_name]['negative'].values[0]
                    if bigger_better:
                        conf = np.max([negative_2cl, negative])
                    elif neg_4cl:
                        conf = negative
                    else:
                        conf = negative_2cl

                    sub_image = sub_image.append(
                        {
                            "id": img_name + "_image",
                            "PredictionString": "none "
                            + str(conf)
                            + " 0 0 1 1",
                        },
                        ignore_index=True,
                    )
        return sub_image


class Generator(keras.utils.Sequence):
    def __init__(
        self,
        df,
        img_size,
        jpg_path,
        metadata_path,
        jpg = False,
        png=True,
        img_from_folder=True,
        batch_size=1,
        n_inputs=2,
    ):
        self.df = df.reset_index(drop=True)
        self.batch_size = batch_size
        self.img_size = img_size
        self.jpg_path = jpg_path
        self.metadata_path = metadata_path
        self.img_from_folder = img_from_folder
        self.n_inputs = n_inputs
        self.jpg = jpg
        self.png = png

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

    def make_img(self, img_path, img_type):
        img_name = img_path.split("/")[-1].split(".dcm")[0]
        if self.img_from_folder:
            try:
                if self.png:
                    img = tf.io.read_file(self.jpg_path + img_name + ".png")
                    img = tf.image.decode_png(img, channels=3)
                elif self.jpg:
                    img = tf.io.read_file(self.jpg_path + img_name + ".jpg")
                    img = tf.image.decode_jpeg(img, channels=3)
                img = tf.image.resize(
                    img,
                    (self.img_size, self.img_size),
                )
                img = tf.cast(img, tf.uint8)
                img = img.numpy()
            except:
                img = self.img_from_dicom(img_path, img_type)
        else:
            img = self.img_from_dicom(img_path, img_type)

        return img

    def _get_one(self, ix):
        img_name = self.df.loc[ix, "id_image"]
        img_path = self.df.loc[ix, "path"]
        modality = self.df.loc[ix, "modality"]
        PatientSex = self.df.loc[ix, "PatientSex"]
        body_part = self.df.loc[ix, "BodyPartExamined"]
        img_type = self.df.loc[ix, "PhotometricInterpretation"]
        img = self.make_img(img_path, img_type)

        with open(self.metadata_path, "r") as f:
            dict_metadata = json.load(f)
        x = {}
        if self.n_inputs == 2:
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
            x["img"] = img
            x["data"] = np.concatenate([patient_sex_x, body_part_x, modality_x])
        else:
            x["img"] = img
        y = np.zeros(4, dtype="uint8")
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

        return x


