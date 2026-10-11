# Skin Lesion Classification Using Deep Learning

An end-to-end deep learning project for multi-class classification of
dermatoscopic skin lesion images from the HAM10000 dataset.

The project compares a custom CNN trained from scratch with transfer
learning using MobileNetV2 and EfficientNetB0. Because HAM10000 is
highly imbalanced, model performance is evaluated using macro-F1 and
macro-recall alongside accuracy.

## Project Overview

The goal is to classify dermatoscopic images into seven diagnostic
categories:

-   `akiec` --- Actinic keratoses and intraepithelial carcinoma
-   `bcc` --- Basal cell carcinoma
-   `bkl` --- Benign keratosis-like lesions
-   `df` --- Dermatofibroma
-   `mel` --- Melanoma
-   `nv` --- Melanocytic nevi
-   `vasc` --- Vascular lesions

The workflow includes exploratory data analysis, leakage-aware data
splitting, image preprocessing and augmentation, model training,
transfer learning, fine-tuning, validation comparison, confusion-matrix
analysis, and final evaluation on a held-out test set.

## Dataset

This project uses the **HAM10000 (Human Against Machine with 10000
training images)** dataset.

The notebook downloads the dataset from Kaggle using:

``` bash
kaggle datasets download -d kmader/skin-cancer-mnist-ham10000 -p data
```

The dataset is not included in this repository because of its size.

A key characteristic of HAM10000 is that multiple images can belong to
the same lesion. To reduce the risk of data leakage, the train,
validation, and test sets are split by `lesion_id` rather than by
individual image.

## Models

Five model configurations are evaluated:

1.  Custom CNN
2.  MobileNetV2 --- Feature Extraction
3.  MobileNetV2 --- Fine-Tuned
4.  EfficientNetB0 --- Feature Extraction
5.  EfficientNetB0 --- Fine-Tuned

For the transfer-learning models, the pretrained base is first used as a
frozen feature extractor and is then partially unfrozen for fine-tuning.

## Validation Results

  Model                                      Accuracy    Macro F1   Macro Recall
  --------------------------------------- ----------- ----------- --------------
  Custom CNN                                    0.747       0.468          0.447
  MobileNetV2 --- Feature Extraction            0.745       0.460          0.426
  MobileNetV2 --- Fine-Tuned                    0.753       0.473          0.441
  EfficientNetB0 --- Feature Extraction         0.716       0.418          0.362
  **EfficientNetB0 --- Fine-Tuned**         **0.745**   **0.512**      **0.452**

The fine-tuned EfficientNetB0 achieved the highest validation macro-F1
and was selected as the final model.

## Final Test Performance

The selected fine-tuned EfficientNetB0 achieved:

-   **Accuracy:** 0.767
-   **Macro F1:** 0.475
-   **Macro Recall:** 0.422

The model performed particularly well on the majority `nv` class, while
several minority classes remained more difficult to classify. The
difference between accuracy and macro-F1 highlights the importance of
class-balanced metrics for this imbalanced dataset.

## Project Structure

``` text
Skin-Lesion-Classification/
├── Skin_Lesion_Classification.ipynb
├── README.md
├── requirements.txt
├── .gitignore
├── data/       # created locally; not tracked by Git
└── models/     # generated model checkpoints; not tracked by Git
```

## Installation

Create a Python environment and install the required packages:

``` bash
pip install -r requirements.txt
```

The Kaggle command-line tool must also be configured with valid Kaggle
credentials before running the dataset-download cell.

## Running the Project

1.  Clone or download this repository.
2.  Install the dependencies from `requirements.txt`.
3.  Configure Kaggle API access.
4.  Open `Skin_Lesion_Classification.ipynb`.
5.  Run the notebook from the beginning.

The notebook creates the local `data/` and `models/` directories
automatically. Dataset files are downloaded into `data/`, while the best
model checkpoints are saved in `models/`.

## Methods and Tools

-   Python
-   TensorFlow / Keras
-   CNNs
-   Transfer Learning
-   MobileNetV2
-   EfficientNetB0
-   Scikit-learn
-   Pandas and NumPy
-   Matplotlib and Seaborn

## Limitations and Future Work

The dataset is strongly imbalanced, and some minority classes contain
relatively few samples. Future work could explore additional pretrained
architectures, alternative strategies for improving minority-class
performance, and systematic hyperparameter tuning.

## Author

**Mina Tayebi**
