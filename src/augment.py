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

import albumentations
import cv2
import numpy as np


class Hard_Aug:
    def augment_image(img):
        h = img.shape[0]
        w = img.shape[1]
        transform = albumentations.Compose(
            [
                albumentations.OneOf(
                    [
                        albumentations.MotionBlur((3, 5)),
                        albumentations.MedianBlur(blur_limit=5),
                        albumentations.GaussianBlur(blur_limit=(3, 5), sigma_limit=0),
                        albumentations.Blur(blur_limit=(3, 5)),
                    ],
                    p=0.1,
                ),
                albumentations.OneOf(
                    [
                        albumentations.GaussNoise(var_limit=[10, 12], mean=1),
                        albumentations.ImageCompression(
                            quality_lower=85, quality_upper=100, compression_type=1
                        ),
                        albumentations.MultiplicativeNoise(
                            multiplier=(0.95, 1.05), per_channel=False, elementwise=True
                        ),
                        albumentations.Downscale(
                            scale_min=0.85,
                            scale_max=0.99,
                            interpolation=cv2.INTER_LINEAR,
                        ),
                    ],
                    p=0.1,
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
                    p=0.1,
                ),
                albumentations.OneOf(
                    [
                        albumentations.OpticalDistortion(
                            distort_limit=0.3,
                            shift_limit=0.9,
                            border_mode=0,
                        ),
                        albumentations.ElasticTransform(
                            alpha=5.0,
                            sigma=50.0,
                            alpha_affine=7.0,
                            interpolation=cv2.INTER_LINEAR,
                            border_mode=0,
                        ),
                        albumentations.GridDistortion(
                            num_steps=5,
                            distort_limit=0.3,
                            interpolation=cv2.INTER_LINEAR,
                            border_mode=0,
                        ),
                        albumentations.Rotate(
                            limit=(-360, 360),
                            interpolation=cv2.INTER_LINEAR,
                            border_mode=0,
                        ),
                        albumentations.ShiftScaleRotate(
                            shift_limit=0.2,
                            scale_limit=0.2,
                            rotate_limit=360,
                            interpolation=cv2.INTER_LINEAR,
                            border_mode=0,
                        ),
                        albumentations.augmentations.crops.transforms.CropAndPad(
                            px=None,
                            percent=(-0.3, 0.2),
                            pad_mode=0,
                            pad_cval=0,
                            pad_cval_mask=0,
                            keep_size=True,
                            sample_independently=True,
                            interpolation=cv2.INTER_LINEAR,
                        ),
                        albumentations.RandomSizedCrop(
                            min_max_height=(0.8 * h, 0.85 * w),
                            height=h,
                            width=w,
                            w2h_ratio=1.0,
                            interpolation=cv2.INTER_LINEAR,
                        ),
                    ],
                    p=0.2,
                ),
                albumentations.OneOf(
                    [
                        albumentations.HorizontalFlip(),
                        albumentations.VerticalFlip(),
                        albumentations.RandomRotate90(),
                        albumentations.Flip(),
                    ],
                    p=0.1,
                ),
            ]
        )

        return transform(image=img)["image"]


class Aug:
    def augment_image(img):
        h = img.shape[0]
        w = img.shape[1]
        transform = albumentations.Compose(
            [
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
                            scale_min=0.8,
                            scale_max=0.99,
                            interpolation=cv2.INTER_LINEAR,
                        ),
                    ],
                    p=0.2,
                ),
                albumentations.OneOf(
                    [
                        albumentations.RandomBrightnessContrast(
                            brightness_limit=(-0.3, 0.2),
                            contrast_limit=(-0.3, 0.2),
                            brightness_by_max=True,
                        ),
                        albumentations.augmentations.transforms.Sharpen(
                            alpha=(0.05, 0.1), lightness=(0.5, 1.0)
                        ),
                        albumentations.augmentations.transforms.RandomToneCurve(
                            scale=0.05
                        ),
                    ],
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
                            interpolation=cv2.INTER_LINEAR,
                            border_mode=0,
                        ),
                        albumentations.GridDistortion(
                            num_steps=5,
                            distort_limit=0.3,
                            interpolation=cv2.INTER_LINEAR,
                            border_mode=0,
                        ),
                    ],
                    p=0.2,
                ),
                albumentations.OneOf(
                    [
                        albumentations.Rotate(
                            limit=(-10, 10),
                            interpolation=cv2.INTER_LINEAR,
                            border_mode=0,
                        ),
                        albumentations.ShiftScaleRotate(
                            shift_limit=0.1,
                            scale_limit=0.1,
                            rotate_limit=10,
                            interpolation=cv2.INTER_LINEAR,
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
                            interpolation=cv2.INTER_LINEAR,
                        ),
                        albumentations.RandomSizedCrop(
                            min_max_height=(0.9 * h, 0.9 * w),
                            height=h,
                            width=w,
                            w2h_ratio=1.0,
                            interpolation=cv2.INTER_LINEAR,
                        ),
                    ],
                    p=0.3,
                ),
            ]
        )

        return transform(image=img)["image"]

class Transform_Aug:
    def augment_image(img):
        h = img.shape[0]
        w = img.shape[1]
        transform = albumentations.Compose(
            [
                albumentations.OneOf(
                    [
                        albumentations.OpticalDistortion(
                            distort_limit=0.3,
                            shift_limit=0.9,
                            border_mode=0,
                        ),
                        albumentations.ElasticTransform(
                            alpha=5.0,
                            sigma=50.0,
                            alpha_affine=7.0,
                            interpolation=cv2.INTER_LINEAR,
                            border_mode=0,
                        ),
                        albumentations.GridDistortion(
                            num_steps=5,
                            distort_limit=0.3,
                            interpolation=cv2.INTER_LINEAR,
                            border_mode=0,
                        ),
                        albumentations.Rotate(
                            limit=(-360, 360),
                            interpolation=cv2.INTER_LINEAR,
                            border_mode=0,
                        ),
                        albumentations.ShiftScaleRotate(
                            shift_limit=0.2,
                            scale_limit=0.2,
                            rotate_limit=360,
                            interpolation=cv2.INTER_LINEAR,
                            border_mode=0,
                        ),
                        albumentations.augmentations.crops.transforms.CropAndPad(
                            px=None,
                            percent=(-0.3, 0.2),
                            pad_mode=0,
                            pad_cval=0,
                            pad_cval_mask=0,
                            keep_size=True,
                            sample_independently=True,
                            interpolation=cv2.INTER_LINEAR,
                        ),
                        albumentations.RandomSizedCrop(
                            min_max_height=(0.8 * h, 0.85 * w),
                            height=h,
                            width=w,
                            w2h_ratio=1.0,
                            interpolation=cv2.INTER_LINEAR,
                        ),
                    ],
                    p=0.1,
                ),
                albumentations.OneOf(
                    [
                        albumentations.HorizontalFlip(),
                        albumentations.VerticalFlip(),
                        albumentations.RandomRotate90(),
                        albumentations.Flip(),
                    ],
                    p=0.1,
                ),
            ]
        )

        return transform(image=img)["image"]

class Flip_Aug:
    def augment_image(img):
        transform = albumentations.Compose(
            [
                albumentations.OneOf(
                    [
                        albumentations.HorizontalFlip(),
                        albumentations.VerticalFlip(),
                        albumentations.RandomRotate90(),
                        albumentations.Flip(),
                    ],
                    p=0.1,
                ),
            ]
        )

        return transform(image=img)["image"]


class Aug_No_transform:
    def augment_image(img):
        transform = albumentations.Compose(
            [
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
                            scale_min=0.8,
                            scale_max=0.99,
                            interpolation=cv2.INTER_LINEAR,
                        ),
                    ],
                    p=0.2,
                ),
                albumentations.OneOf(
                    [
                        albumentations.RandomBrightnessContrast(
                            brightness_limit=(-0.3, 0.2),
                            contrast_limit=(-0.3, 0.2),
                            brightness_by_max=True,
                        ),
                        albumentations.augmentations.transforms.Sharpen(
                            alpha=(0.05, 0.1), lightness=(0.5, 1.0)
                        ),
                        albumentations.augmentations.transforms.RandomToneCurve(
                            scale=0.05
                        ),
                    ],
                    p=0.2,
                ),
            ]
        )

        return transform(image=img)["image"]


class Aug_Crop:
    def augment_image(img):
        h = img.shape[0]
        w = img.shape[1]
        transform = albumentations.Compose(
            [
                albumentations.OneOf(
                    [
                        albumentations.MotionBlur((3, 5)),
                        albumentations.MedianBlur(blur_limit=5),
                        albumentations.GaussianBlur(blur_limit=(3, 5), sigma_limit=0),
                        albumentations.Blur(blur_limit=(3, 5)),
                    ],
                    p=0.1,
                ),
                albumentations.OneOf(
                    [
                        albumentations.GaussNoise(var_limit=[10, 20], mean=1),
                        albumentations.ImageCompression(
                            quality_lower=85, quality_upper=100, compression_type=1
                        ),
                        albumentations.MultiplicativeNoise(
                            multiplier=(0.95, 1.05), per_channel=False, elementwise=True
                        ),
                        albumentations.Downscale(
                            scale_min=0.85,
                            scale_max=0.99,
                            interpolation=cv2.INTER_LINEAR,
                        ),
                    ],
                    p=0.1,
                ),
                albumentations.OneOf(
                    [
                        albumentations.RandomBrightnessContrast(
                            brightness_limit=(-0.1, 0.2),
                            contrast_limit=(-0.1, 0.2),
                            brightness_by_max=True,
                        ),
                        albumentations.augmentations.transforms.Sharpen(
                            alpha=(0.05, 0.1), lightness=(0.5, 1.0)
                        ),
                        albumentations.augmentations.transforms.RandomToneCurve(
                            scale=0.05
                        ),
                    ],
                    p=0.1,
                ),
                albumentations.OneOf(
                    [
                        albumentations.OpticalDistortion(
                            distort_limit=0.1,
                            shift_limit=0.1,
                            border_mode=1,
                        ),
                        albumentations.ElasticTransform(
                            alpha=2.0,
                            sigma=2.0,
                            alpha_affine=2.0,
                            interpolation=cv2.INTER_LINEAR,
                            border_mode=1,
                        ),
                        albumentations.GridDistortion(
                            num_steps=5,
                            distort_limit=0.1,
                            interpolation=cv2.INTER_LINEAR,
                            border_mode=1,
                        ),
                    ],
                    p=0.1,
                ),
                albumentations.OneOf(
                    [
                        albumentations.augmentations.crops.transforms.CropAndPad(
                            px=None,
                            percent=(-0.01, 0.05),
                            pad_mode=1,
                            pad_cval=0,
                            pad_cval_mask=0,
                            keep_size=True,
                            sample_independently=True,
                            interpolation=cv2.INTER_LINEAR,
                        ),
                        albumentations.RandomSizedCrop(
                            min_max_height=(0.95 * h, 0.95 * w),
                            height=h,
                            width=w,
                            w2h_ratio=1.0,
                            interpolation=cv2.INTER_LINEAR,
                        ),
                        albumentations.HorizontalFlip(),
                        albumentations.VerticalFlip(),
                        albumentations.RandomRotate90(),
                    ],
                    p=0.1,
                ),
                albumentations.CoarseDropout(
                    max_holes=10,
                    max_height=36,
                    max_width=36,
                    min_holes=1,
                    min_height=6,
                    min_width=6,
                    fill_value=0,
                    p=0.1
                ),
            ]
        )

        return transform(image=img)["image"]