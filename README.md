# [SIIM-FISABIO-RSNA COVID-19 Detection](#https://www.kaggle.com/c/siim-covid19-detection/overview)
In this competition task is to identify and localize COVID-19 abnormalities on chest radiographs.

Methods:
 - classifiation using EfficientNetB7,
 - lung detection for cropping lungs from images with EfficientNetB0,
 - opacity detection using Scaled-YOLOv4 and [YOLOv5](#https://github.com/ultralytics/yolov5)
 - postprocessing bounding boxes using Weighted boxes fusion

Libraries:
pandas, numpy, cv2, PIL, albumentations, pydicom, tensorflow, tensorflow_addons, matplotlib, sklearn, ast, os, re, tqdm, shutil, json, collections, 






# [Weighted boxes fusion](#https://github.com/ZFTurbo/Weighted-Boxes-Fusion)
```
@article{solovyev2021weighted,
  title={Weighted boxes fusion: Ensembling boxes from different object detection models},
  author={Solovyev, Roman and Wang, Weimin and Gabruseva, Tatiana},
  journal={Image and Vision Computing},
  pages={1-6},
  year={2021},
  publisher={Elsevier}
}
```
# [Scaled-YOLOv4](#https://github.com/WongKinYiu/ScaledYOLOv4)
```
@InProceedings{Wang_2021_CVPR,
    author    = {Wang, Chien-Yao and Bochkovskiy, Alexey and Liao, Hong-Yuan Mark},
    title     = {{Scaled-YOLOv4}: Scaling Cross Stage Partial Network},
    booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
    month     = {June},
    year      = {2021},
    pages     = {13029-13038}
}
```

# [Object-Detection-Metrics](#https://github.com/rafaelpadilla/Object-Detection-Metrics)
```
@Article{electronics10030279,
AUTHOR = {Padilla, Rafael and Passos, Wesley L. and Dias, Thadeu L. B. and Netto, Sergio L. and da Silva, Eduardo A. B.},
TITLE = {A Comparative Analysis of Object Detection Metrics with a Companion Open-Source Toolkit},
JOURNAL = {Electronics},
VOLUME = {10},
YEAR = {2021},
NUMBER = {3},
ARTICLE-NUMBER = {279},
URL = {https://www.mdpi.com/2079-9292/10/3/279},
ISSN = {2079-9292},
DOI = {10.3390/electronics10030279}
}
```
