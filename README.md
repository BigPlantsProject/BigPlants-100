<div align="center"> <h1>From Classical CNNs to Modern Deep Architectures: <br>Multiclass Plant Recognition on the BigPlants Dataset🌿</h1> </div>

## 📚 Overview

Plant image classification remains a challenging task due to variations in lighting, complex backgrounds, and high visual similarity among species. This task is crucial for biodiversity research and medicinal plant identification. In this study, we present **BigPlants-100**, a newly curated dataset consisting of 100 plant species selected from the Vietnam Plant Database based on their medicinal and toxic importance. Using this dataset, we evaluated several state-of-the-art architectures, including ConvNeXt V2-Tiny, EfficientNetV2-S, and ResNet-50. Furthermore, we implemented a multi-teacher knowledge distillation framework to guide a MobileNetV3-Large student network. Experimental results demonstrate that the distilled student model achieves notable performance gains, **improving both accuracy and macro F1-score from 0.849 to 0.894**, while **reducing training loss by more than 31%** compared to individually trained models. These findings highlight the effectiveness of multi-teacher knowledge distillation in enhancing generalization and accuracy for large-scale multi-class plant image classification.

## 📁 Source Code Structure

```
BigPlants-100/
├─ standalone/
│  ├─ convnextv2s/
│  │  ├─ output/
│  │  │  ├─ convnextv2_test_classification_report.csv
│  │  │  ├─ ...
│  │  └─ convnextv2_standalone.py
│  ├─ efficientnetv2s/
│  │  ├─ output/
│  │  │  ├─ efficientnetv2s_test_classification_report.csv
│  │  │  ├─ ...
│  │  └─ efficientnetv2s_standalone.py
│  ├─ mobilenetv3large/
│  │  ├─ output/
│  │  │  ├─ mobilenetv3large_test_classification_report.csv
│  │  │  ├─ ...
│  │  └─ mobilenetv3large_standalone.py
│  └─ resnet50/
│  │  ├─ output/
│  │  │  ├─ resnet50_test_classification_report.csv
│  │  │  ├─ ...
│  │  └─ resnet50_standalone.py
├─ cross_validation/
│  ├─ convnextv2s/
│  │  ├─ output/
│  │  │  ├─ convnextv2_test_classification_report_fold1.csv
│  │  │  ├─ ...
│  │  └─ convnextv2_cross_validation.py
│  ├─ efficientnetv2s/
│  │  ├─ output/
│  │  │  ├─ efficientnetv2s_test_classification_report_fold1.csv
│  │  │  ├─ ...
│  │  └─ efficientnetv2s_cross_validation.py
│  ├─ mobilenetv3large/
│  │  ├─ output/
│  │  │  ├─ mobilenetv3large_test_classification_report_fold1.csv
│  │  │  ├─ ...
│  │  └─ mobilenetv3large_cross_validation.py
│  └─ resnet50/
│  │  ├─ output/
│  │  │  ├─ resnet50_test_classification_report_fold1.csv
│  │  │  ├─ ...
│  │  └─ resnet50_cross_validation.py
├─ knowledge_distillation/
│  ├─ multi_teacher_kd.py
│  ├─ student_test_report.csv
│  ├─ kd_student_test_confusion_matrix.npy
│  └─ ...
└─ bigplants100_name_list.txt
└─ check_duplicates_phash.py
└─ preprocessing_dataset.py
└─ README.md
```

## 🌳 BigPlants-100 Dataset

- The entire raw dataset is available at the following link:
  ```
  https://drive.google.com/drive/folders/1zbczeI8HnfzKhMAybibRq9a40Jcm7bX_?usp=sharing
  ```
- The full dataset is available at the following link:
  ```
  https://drive.google.com/drive/folders/1uEFtoS-XivF030a5BAbM8mD341eqd_I9?usp=sharing
  ```
