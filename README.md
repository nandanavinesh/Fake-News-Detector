# Fake News Detection using Machine Learning

## Overview
This project detects whether a news article is **Fake** or **Real** using Natural Language Processing (NLP) and Machine Learning.

The model was trained on the **ISOT Fake News Dataset** and achieved **~99% accuracy**.

---

## Workflow
1. Data loading (Fake & Real news datasets)
2. Data cleaning & preprocessing
3. Feature engineering (text-based + metadata features)
4. TF-IDF vectorization (with n-grams)
5. Train-test split
6. Model training using Passive Aggressive Classifier
7. Model evaluation (Accuracy + Confusion Matrix)

---

## Feature Engineering
- Combined **title + article text**
- Text length features
- TF-IDF with **unigrams & bigrams**
- Stopword removal

---
## Dataset
This project uses the **ISOT Fake News Dataset**.

Due to GitHub file size limitations, the dataset files are not included in this repository.

You can download the dataset from Kaggle:
https://www.kaggle.com/datasets/clmentbisaillon/fake-and-real-news-dataset

After downloading, place the following files in the project directory:
- `Fake.csv`
- `True.csv`

---

## Technologies Used
- Python
- Pandas
- NumPy
- Scikit-learn
- SciPy

---

## Results
- Accuracy: **~99%**
- Strong class separation shown in confusion matrix

---

## How to Run
```bash
pip install -r requirements.txt
python main.py
