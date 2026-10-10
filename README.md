# 🫁 Pulmonary Tuberculosis Detection System

An automated Computer-Aided Diagnostic (CAD) system utilizing **Transfer Learning with VGG16** and **TensorFlow** to detect Tuberculosis (TB) from chest X-ray radiographs, paired with a modern, responsive **Flask** web application.

---

## 👥 Team Members

| Name | Registration Number |
| :--- | :--- |
| **Ishan Choudhary** | 25BCE10753 |
| **Khush Gupta** | 25BCE10038 |
| **Vidit Choudhary** | 25BCE10749 |
| **Saransh Mathur** | 25BCE10354 |
| **Somya Bhardwaj** | 25BCE10409 |

---

## 🛠 Tech Stack

* **Programming Language:** Python 3.10+
* **Deep Learning Framework:** TensorFlow / Keras (VGG16 Transfer Learning)
* **Backend Web Framework:** Flask
* **Frontend:** HTML5, CSS3, JavaScript (interactive image preview & real-time progress bars)
* **Image Processing:** Pillow (PIL), NumPy

---

## 📊 Dataset & Splitting

The project utilizes the benchmark **Tuberculosis (TB) Chest X-ray Database** containing **4,200 chest radiographs** (3,500 Normal and 700 TB positive).

The dataset is partitioned into:
* **Training Set (70%):** 2,939 images
* **Validation Set (15%):** 630 images
* **Held-out Test Set (15%):** 631 images (strictly unseen)

---

## 🏆 Model Performance (On 631 Unseen Test Scans)

| Evaluation Metric | Score | Clinical Relevance |
| :--- | :--- | :--- |
| **Overall Accuracy** | **98.57%** | 622 of 631 test images correctly diagnosed |
| **Sensitivity (Recall)** | **93.40%** | Caught 99 of 106 positive TB patients |
| **Specificity** | **99.62%** | Cleared 523 of 525 healthy lungs |

### Confusion Matrix:
```text
                   Predicted Normal    Predicted TB
Actual Normal            523 (TN)            2 (FP)
Actual TB                  7 (FN)           99 (TP)
```

---

## 🚀 How to Run the Project

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Train the Model *(Optional - pretrained `tb_model.h5` already included)*
```bash
python train_model.py
```

### 3. Check Model Accuracy & Evaluation
```bash
python calculate_accuracy.py
```

### 4. Launch the Web Application
```bash
python app.py
```
Open your browser and navigate to:
```text
http://127.0.0.1:5000
```
Upload any chest X-ray image from the `dataset/test/` folder to view the real-time prediction and confidence score.
