<div align="center"> <h1>From Classical CNNs to Modern Deep Architectures: <br>Multiclass Plant Recognition on the BigPlants Dataset🌿</h1> </div>

## 📚 Overview

Plant image classification remains a challenging task due to variations in lighting, complex backgrounds, and high visual similarity among species. This task is crucial for biodiversity research and medicinal plant identification. In this study, we present **BigPlants-100**, a newly curated dataset consisting of 100 plant species selected from the Vietnam Plant Database based on their medicinal and toxic importance. Using this dataset, we evaluated several state-of-the-art architectures, including ConvNeXt V2-Tiny, EfficientNetV2-S, ResNet-50 and ViT-B/16. Furthermore, we implemented a multi-teacher knowledge distillation framework to guide a MobileNetV3-Large student network. Experimental results demonstrate that the distilled student model achieves notable performance gains, **improving both accuracy and macro F1-score from 0.849 to 0.894**, while **reducing training loss by more than 31%** compared to individually trained models. These findings highlight the effectiveness of multi-teacher knowledge distillation in enhancing generalization and accuracy for large-scale multi-class plant image classification.

## 🌳 Dataset
**The full dataset is available at**: **[BigPlants-100 Dataset](https://drive.google.com/drive/folders/1uEFtoS-XivF030a5BAbM8mD341eqd_I9?usp=sharing)**

## 📄 License & Terms of Use

The **BigPlants-100** dataset is made available strictly for non-commercial academic research and educational purposes. 

By downloading, accessing, or using this dataset, you agree to comply with the terms and conditions outlined in the **[LICENSE.md](./LICENSE.md)** agreement. Commercial use, redistribution, and uncredited usage are strictly prohibited.
