# Sleep Stage Classification from EEG

Traditional Machine Learning & Deep Learning Approaches

This exercise session explores two complete pipelines for **automatic sleep stage classification** using EEG recordings:

* **Exercise 1 — Traditional ML (Frequency Domain)**
* **Exercise 2 — Deep Learning (Time Domain)**

Both frameworks operate on the same dataset of single-channel EEG (Fpz-Cz), segmented into **30-second epochs**.

---

## Data Overview

Each subject provides:

* **Raw EEG signal**

  * 100 Hz sampling
  * 3000 timepoints per epoch
  * stored in `data/signals/subject_i.pt`

* **Sleep stage labels**

  * Classes: Wake, N1, N2, N3, REM (encoded 0–4)
  * stored in `data/stages/subject_i.pt`

* **PSD features (Exercise 1)**

  * Power spectral density across ~500 frequency bins
  * stored in `data/features/subject_i.pt`

---

# Exercise 1 — Traditional Machine Learning

### **1. Pipeline**

Work in the **frequency domain** using PSD features:

```
PSD → power_scale (dB) → StandardScaler → (band features / PCA) → classifier
```

We tested two feature engineering strategies:

* **Band-power averaging** (delta, theta, alpha, sigma, beta)
* **PCA** with components ∈ {2,4,8,16,32,64,128,256}

Classifiers evaluated with **Leave-One-Subject-Out (LOSO)**:

* KNN
* RandomForest
* LogisticRegression
* SVM (RBF)
* DecisionTree

### **2. Best Performing Model**

```
SVC (RBF) + PCA(128)  
→ Accuracy ≈ 75% (LOSO)
```

### **3. Deployment**

The best pipeline is retrained on training subjects, saved as:

```
best_sklearn_model.pkl
```

and evaluated on unseen subjects (11–15) using:

* normalized confusion matrix
* classification report

---

# Exercise 2 — Deep Learning (Raw EEG)

### **1. Pipeline**

Work in the **time domain** using raw signals:

```
Raw EEG (3000 samples) → StandardScaler → SimpleEEGNet (1D CNN)
```

Key aspects:

* Per-subject LOSO cross-validation
* Two fixed validation subjects for early stopping
* CNN learns temporal patterns directly (no manual features)

### **2. Model**

A lightweight 1D-CNN:

```
Conv1D → BatchNorm → ReLU → MaxPool  
(repeated 3 times)  
→ Flatten → Linear(5 classes)
```

Implemented with **skorch** for sklearn-style training.

### **3. Evaluation**

For each LOSO fold:

* Train on N–3 subjects
* Validate on fixed validation subjects
* Test on the held-out subject

Metrics: accuracy, f1_weighted, balanced accuracy, precision_weighted, recall_weighted.

---

# Visualization & Diagnostics

Both exercises include tools to plot:

* raw EEG epochs
* frequency-band overlays
* LOSO confusion matrices
* classification reports
* PCA variance curves
* per-classifier accuracy across PCA dimensions

---

# Summary

* **Traditional ML** provides interpretable baselines, especially with PCA or band-power features.
* **Deep learning** learns directly from raw EEG and avoids manual feature crafting.
* Both pipelines enforce strict subject-level separation via LOSO, preventing leakage and ensuring realistic generalization.

