# 🌿From Classical CNNs to Modern Deep Architectures: Multiclass Plant Recognition on the BigPlants Dataset

## 📚 Overview

Plant image classification remains a challenging task due to variations in lighting conditions, complex backgrounds, and high visual similarity among species. This task plays a crucial role in biodiversity research, medicinal plant identification, and ecological monitoring. In this study, we present BigPlants-100, a newly curated dataset consisting of images from 100 plant species, each representing a distinct class, selected from the Vietnam Plant Database. All images were manually collected and annotated by our team from reliable sources. The selected species were chosen based on their medicinal importance, toxicity, or both. Using the BigPlants-100 dataset, we conducted a comprehensive evaluation of several state-of-the-art deep learning architectures, including ConvNeXtV2-S, EfficientNetV2-S, MobileNetV3-Large, and ResNet-50, employing cross-validation to ensure result stability and reliability. Furthermore, we implemented a multi-teacher knowledge distillation framework, where multiple high-performing models act as teachers to guide a student network. Experimental results demonstrate that using MobileNetV3-Large as the student distilled from ConvNeXtV2-S, EfficientNetV2-S and ResNet-50 yields superior performance compared to individually trained models. These findings highlight the effectiveness of multi-teacher knowledge distillation in enhancing generalization and accuracy for large-scale multi-class plant image classification.

## 📁 Source Code Structure

```
BigPlants-100/
├─ standalone/
│  ├─ convnextv2s/
│  │  ├─ output/
│  │  │  ├─ convnextv2_test_classification_report.csv
│  │  │  ├─ ...
│  │  └─ convnextv2s_standalone.py
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
│  │  └─ convnextv2s_cross_validation.py
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
│  ├─ best_student_kd.pt
│  ├─ student_test_report.csv
│  ├─ kd_student_test_confusion_matrix.npy
│  └─ ...
└─ bigplants100_name_list.csv
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
