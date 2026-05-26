# 🩻 Chest X-Ray Pneumonia Detection

## 📌 Overview
This project aims to detect pneumonia from chest X-ray images using **Convolutional Neural Networks (CNNs)** and **Transfer Learning** with **ResNet50**. The model classifies images into two categories: `NORMAL` and `PNEUMONIA`.

The dataset used is the [Chest X-Ray Images (Pneumonia)](https://www.kaggle.com/paultimothymooney/chest-xray-pneumonia) dataset from Kaggle.

## 🧠 Models Implemented
1. **Custom CNN** – Built from scratch using multiple Conv2D + MaxPooling layers.
2. **ResNet50 (Transfer Learning)** – Pre-trained on ImageNet, fine-tuned for pneumonia classification.

## 📊 Key Results
| Model       | Test Accuracy | Precision (Pneumonia) | Recall (Pneumonia) | F1-Score |
|-------------|---------------|----------------------|--------------------|-----------|
| Custom CNN  | 86%           | 0.84                 | 0.97               | 0.90      |
| ResNet50    | 89.4%         | 0.90                 | 0.93               | 0.92      |

- **Class imbalance** handled using `class_weight` in training.
- **Data augmentation** (rotation, zoom, flip) applied to improve generalization.
- **Validation split** (20% of training data) used for early stopping and learning rate scheduling.

## 🛠️ Tech Stack
- Python 3.x
- TensorFlow / Keras
- ResNet50 (pre-trained)
- OpenCV, Matplotlib, Seaborn
- Scikit-learn (metrics, class weights)

## 📁 Dataset Structure
chest_xray/
├── train/
│ ├── NORMAL/
│ └── PNEUMONIA/
├── val/
│ ├── NORMAL/
│ └── PNEUMONIA/
└── test/
├── NORMAL/
└── PNEUMONIA/

## 🚀 How to Run
1. Clone the repository.
2. Download the dataset from [Kaggle](https://www.kaggle.com/paultimothymooney/chest-xray-pneumonia).
3. Update the dataset path in the notebook.
4. Run all cells in `X_ray_classification_PNEUMONIA.ipynb`.

## 📈 Results Visualization
- Training vs validation accuracy/loss curves
- Confusion matrix
- Sample predictions with images

## 📌 Future Improvements
- Experiment with other pre-trained models (VGG16, DenseNet)
- Deploy as a web app using Flask or FastAPI
- Add Grad-CAM visualizations for model interpretability

## 📄 License
This project is for educational purposes only.
