# Digit Recognition Using K-Nearest Neighbors (KNN)

This project focuses on building a handwritten digit classification system using the K-Nearest Neighbors (KNN) algorithm entirely from scratch. No prebuilt classifiers or advanced ML libraries were used. Instead, the focus was on manually constructing each step of the pipeline using fundamental tools like NumPy, pandas, and matplotlib.

---

## Introduction

The task was to classify handwritten digits (0–9) from the MNIST dataset using KNN, one of the most intuitive algorithms in machine learning. While often overshadowed by more complex models today, KNN is still an excellent tool for understanding classification, distance metrics, and the importance of data preprocessing.

We approached this project by designing a full machine learning pipeline: from data loading and visualization, to implementing KNN manually, to evaluating performance using accuracy, confusion matrices, and visual inspection.

---


## Dataset

We used the **MNIST dataset**, containing 70,000 grayscale images of handwritten digits — 60,000 for training and 10,000 for testing. Each image is 28x28 pixels, resulting in 784 features once flattened into a one-dimensional array. Pixel intensity values range from 0 (black) to 255 (white).

# To load the dataset:
* from sklearn.datasets import fetch_openml

## Preprocessing

Before feeding the data into our KNN model, two critical preprocessing steps were applied:

* **Flattening**
  The 28x28 pixel images were reshaped into 784-length vectors. This transformation was necessary to treat each image as a point in high-dimensional space.

* **Normalization**
  Pixel values were originally in the range \[0, 255]. We scaled them to \[0, 1] by dividing all values by 255.0. This ensured that features contributed equally to distance calculations.

---

## Visualization

We visualized several digits using matplotlib by reshaping the vectors back to their original 28x28 form. This not only helped verify the correctness of the data but also gave a sense of the variability in handwriting styles.

These visual samples provided context and confirmed that the dataset had sufficient complexity and real-world variation to justify the modeling efforts.



## PCA for Dimensionality Reduction

To improve the performance of the KNN algorithm (which suffers in high dimensions), we implemented **Principal Component Analysis (PCA)** using NumPy. The goal was to retain 95% of the original variance while reducing dimensionality, which significantly reduced the time taken to compute distances during KNN classification.

This optional step also helped in visualizing the data and removing redundant or noisy features that could harm performance.



## Dataset Splitting

To support training and evaluation:

* **60%** of the data was used for training
* **20%** for validation (to select the best value of `k`)
* **20%** for final testing

However, due to the dataset's large size and the computational cost of KNN, we worked with **a random subset** of the data to speed up training and testing while maintaining statistical relevance.

---

## KNN Implementation

The heart of the project is the manual implementation of the K-Nearest Neighbors classifier using only NumPy — no `sklearn.neighbors`.

The algorithm performs as follows:

1. For a given test image, compute distances (Euclidean) to all training images.
2. Select the `k` nearest neighbors.
3. Perform a majority vote to assign a label.
4. Repeat for all test images.

We made use of **vectorized operations** in NumPy for efficient distance computation and prediction.

---

## Tuning and Validation

We tested multiple values of `k` (odd numbers from 1 to 15) to find the optimal number of neighbors. Odd values help avoid tie scenarios during voting.

### Observations:

* **Highest validation accuracy (\~92.6%)** was achieved at `k = 1` and `k = 3`.
* Accuracy declined gradually as `k` increased, likely due to over-smoothing and underfitting.
* **Final choice: `k = 3`**, striking a balance between capturing local details and generalization.

A line graph was plotted to visualize accuracy against various `k` values.

---

## Evaluation

The model was evaluated on the test set using:

* **Accuracy**
* **Confusion Matrix**
* **Classification Report (Precision, Recall, F1-score)**

### Key Findings:

* **Overall test accuracy: \~91%**
* Strong performance on digits like 0, 1, and 6.
* Slight confusion between visually similar digits (e.g., 2 and 5).
* Precision and recall were generally above 0.90 across most classes.

The confusion matrix clearly illustrated where misclassifications occurred, highlighting the need for more sophisticated techniques if perfect classification is desired.

---

## Final Results

 Metric                | Score                        

 Accuracy              |  \~91%                        
 Best `k`              |  3                            
 Preprocessing         |  Normalization + Optional PCA 
 PCA Variance Retained |  95%                          




.

