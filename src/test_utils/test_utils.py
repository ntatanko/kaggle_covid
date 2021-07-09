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
import os
import re
import cv2
import json
import shutil
import numpy as np
import pandas as pd
from PIL import Image
import pydicom as dicom
import tensorflow as tf
from tensorflow import keras
from tqdm import tqdm

# -

class Test:
    def crop(img, model):
        img = np.stack([img, img, img], axis=-1)
        img_224 = tf.image.resize(
            img,
            (224, 224),
        )
        img_224 = tf.cast(img_224, tf.uint8)
        img_224 = tf.expand_dims(img_224, axis=0)
        coord = model.predict(img_224)[0]
        xmin = coord[0] * img.shape[1]
        ymin = coord[1] * img.shape[0]
        xmax = coord[2] * img.shape[1]
        ymax = coord[3] * img.shape[0]
        return xmin, ymin, xmax, ymax

    def prepare_data(
        data_path,
        path_to_save,
        resized_yolo_path = None,
        norm=False,
        classification_img_path=None,
        train_mode=True,
        n_train_sample=10,
        crop_model_path=None,
        img_size=600,
        yolo_img_size = 1024,
        npy_yolo = True
    ):
        if crop_model_path is not None:
            crop_model = keras.models.load_model(crop_model_path)
        if classification_img_path is not None:
            if not os.path.exists(classification_img_path):
                os.makedirs(classification_img_path)

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
                    test_df.loc[img_file, "image"] = img_file[:-4]
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
                    # image
                    img = data_file.pixel_array
                    if data_file.PhotometricInterpretation == "MONOCHROME1":
                        img = img.max() - img
                    img = (img - img.min()) / (img.max() - img.min())
                    img = (np.array(img) * 255).astype("uint8")
                    # images for classification
                    if crop_model_path is not None:
                        xmin, ymin, xmax, ymax = Test.crop(img, crop_model)
                        test_df.loc[img_file, "xmin"] = xmin
                        test_df.loc[img_file, "ymin"] = ymin
                        test_df.loc[img_file, "xmax"] = xmax
                        test_df.loc[img_file, "ymax"] = ymax
                        # cropped image
                        img_cl = img[int(ymin) : int(ymax), int(xmin) : int(xmax)]
                        img_cl = np.stack([img_cl, img_cl, img_cl], axis=-1)
                        img_cl = tf.image.resize(
                                img_cl,
                                (img_size, img_size),
                            )
                    else:
                        img_cl = np.stack([img, img, img], axis=-1)
                        img_cl = tf.image.resize(
                                img_cl,
                                (img_size, img_size),
                            )  
                    if norm:
                        img_cl = tf.cast(img_cl, tf.float32) / 255.0
                    else:
                        img_cl = tf.cast(img_cl, tf.uint8)
                    np.save(classification_img_path + img_file[:-4], img_cl[:,:,0])
                    
                    # images for object detection
                    img = np.stack([img, img, img], axis=-1)
                    if resized_yolo_path is not None:
                        if not os.path.exists(resized_yolo_path):
                            os.makedirs(resized_yolo_path)
                        h0, w0 = img.shape[:2]
                        r = yolo_img_size / max(h0, w0)
                        img_yolo = cv2.resize(img, (int(w0 * r), int(h0 * r)), interpolation=cv2.INTER_LINEAR)
                        if npy_yolo:
                            np.save(resized_yolo_path + img_file[:-4], img_yolo[:,:,0])
                        else:
                            img_yolo = Image.fromarray(img_yolo)
                            img_yolo.save(resized_yolo_path + img_file[:-4] + ".png", format="PNG")
                    else:
                        if not os.path.exists(path_to_save):
                            os.mkdir(path_to_save)
                        if npy_yolo:
                            np.save(path_to_save + img_file[:-4], img[:,:,0])
                        else:
                            img = Image.fromarray(img)
                            img.save(path_to_save + img_file[:-4] + ".jpg", format="JPEG")

        test_df = test_df.reset_index(drop=True)
        test_df["path"] = test_df["path"].str.replace("\../", "/kaggle/")
        return test_df

    def make_classification(
        test_df_,
        model_path,
        model_2cl_path,
        generator,
        obj_det=True,
        classification=True,
        max_value = True
    ):
        pred_df = pd.DataFrame(columns=["image","negative", "typical", "indeterminate", "atypical", 'model'])
        sub_study = pd.DataFrame(columns=["id", "PredictionString"])
        labels = ["negative", "typical", "indeterminate", "atypical"]
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
            pred = [model.predict(generator) for model in eff_models]
            pred_mean = np.mean(pred, axis=0)
            for i in range(len(eff_models)):
                for j in range(len(pred[i])):
                    img_name = test_df_.loc[j, 'image']
                    pred_df = pred_df.append({"image":img_name,"negative":pred[i][j][0], "typical":pred[i][j][1], "indeterminate":pred[i][j][2], "atypical":pred[i][j][3], 'model':i}, ignore_index=True)
            test_df_[["negative", "typical", "indeterminate", "atypical"]] = pred_mean
            if model_2cl_path is not None:
                pred_neg = np.mean([m.predict(generator) for m in models_2cl], axis=0)
                test_df_["negative_2cl"] = pred_neg[:, 0]

            # PredictionString in format 'label threshold 0 0 1 1'
            if max_value:
                groupped_df = test_df_.groupby("StudyInstanceUID")[
                ["negative", "typical", "indeterminate", "atypical"]
            ].max()
            else:
                groupped_df = test_df_.groupby("StudyInstanceUID")[
                ["negative", "typical", "indeterminate", "atypical"]
            ].mean()
            for ix in groupped_df.index.tolist():
                predictions = list(map(str, (np.round(groupped_df.loc[ix].values, 8))))
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
            for img_name in test_df_.image.tolist():
                sub_image = sub_image.append(
                    {"id": img_name + "_image", "PredictionString": "none 1 0 0 1 1"},
                    ignore_index=True,
                )
            return sub_study, sub_image, test_df_, pred_df
        return sub_study, test_df_, pred_df

    def make_predictions(
        test_df_,
        model_path,
        generator,
        neg_cl=False
    ):
        models = []
        for m_path in model_path:
            if ".h5" in m_path:
                models.append(keras.models.load_model(m_path))
            else:
                for file in os.listdir(m_path):
                    if ".h5" in file:
                        models.append(keras.models.load_model(m_path + file))
        
        pred = [model.predict(generator) for model in models]
        if neg_cl:
            pred_df = pd.DataFrame(columns=["image","negative",'model'])
        else:
             pred_df = pd.DataFrame(columns=["image","negative", "typical", "indeterminate", "atypical", 'model'])
        for i in range(len(models)):
            for j in range(len(pred[i])):
                img_name = test_df_.loc[j, 'image']
                if neg_cl:
                    pred_df = pred_df.append({"image":img_name,"negative":pred[i][j][0],'model':i}, ignore_index=True)
                else:
                    pred_df = pred_df.append({"image":img_name,"negative":pred[i][j][0], "typical":pred[i][j][1], "indeterminate":pred[i][j][2], "atypical":pred[i][j][3], 'model':i}, ignore_index=True)

        return pred_df

    def make_bbox_df(
        test_df_,
        SAVE_BBOX_PATH,
        bigger_better=True,
        neg_4cl=False,
        conf_1=False,
    ):
        sub_image = pd.DataFrame(columns=["id", "PredictionString"])
        for file in os.listdir(SAVE_BBOX_PATH):
            if ".txt" in file:
                img_name = file[:-4]
                w = test_df_[test_df_["image"] == img_name]["width"].values[0]
                h = test_df_[test_df_["image"] == img_name]["height"].values[0]
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
        for img_name in test_df_.image.tolist():
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
                    if "negative_2cl" in test_df_.columns:
                        negative_2cl = test_df_[test_df_["image"] == img_name][
                            "negative_2cl"
                        ].values[0]
                    else:
                        negative_2cl = test_df_[test_df_["image"] == img_name][
                            "negative"
                        ].values[0]
                    negative = test_df_[test_df_["image"] == img_name][
                        "negative"
                    ].values[0]
                    if neg_4cl:
                        conf = negative
                    if bigger_better:
                        conf = np.max([negative_2cl, negative])
                    else:
                        conf = negative_2cl

                    sub_image = sub_image.append(
                        {
                            "id": img_name + "_image",
                            "PredictionString": "none " + str(conf) + " 0 0 1 1",
                        },
                        ignore_index=True,
                    )
        return sub_image


class Generator(keras.utils.Sequence):
    def __init__(
        self,
        df,
        img_size,
        cache_path,
        batch_size=1,
        norm=False,
        crop = True
    ):
        self.df = df.reset_index(drop=True)
        self.batch_size = batch_size
        self.img_size = img_size
        self.norm = norm
        self.crop = crop
        self.cache_path = cache_path

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
            xmin, ymin, xmax, ymax = coord
            img = img[int(ymin) : int(ymax), int(xmin) : int(xmax)]
        img = np.stack([img, img, img], axis=-1)
        img = tf.image.resize(
            img,
            (self.img_size, self.img_size),
        )
        if self.norm:
            img = tf.cast(img, tf.float32) / 255.0
        else:
            img = tf.cast(img, tf.uint8)
        return img.numpy()

    def _get_one(self, ix):
        img_name = self.df.loc[ix, "image"]
        img_path = self.df.loc[ix, "path"]
        modality = self.df.loc[ix, "modality"]
        img_type = self.df.loc[ix, "PhotometricInterpretation"]
        if self.crop:
            coord = self.df.loc[ix, ['xmin', 'ymin', 'xmax', 'ymax']].values
        else:
            coord = None
        try:
            img = np.load(self.cache_path+img_name+'.npy')
            img = np.stack([img, img, img], axis=-1)
        except:
            img = self.img_from_dicom(img_path, img_type, coord)
        x = img
        return x

    def __getitem__(self, batch_ix):
        x_ = []
        for i in range(self.batch_size):
            xx = self._get_one(i + self.batch_size * batch_ix)
            x_.append(xx)
        x = np.array(x_)
        return x

