# 📧 Spam Email Classification

![Python](https://img.shields.io/badge/Python-3.x-blue?logo=python)
![Scikit-learn](https://img.shields.io/badge/Scikit--learn-Model-orange?logo=scikit-learn)
![License: MIT](https://img.shields.io/badge/License-MIT-green)
![Status](https://img.shields.io/badge/Status-Completed-success)

## 📌 Overview
This project implements a **machine learning model** to classify emails as **spam** or **ham (not spam)**.  
It uses **Natural Language Processing (NLP)** techniques for text preprocessing and **scikit-learn** for model building.

## ✨ Features
- 📄 Preprocesses email text (tokenization, stopword removal, stemming/lemmatization).
- 🔍 Converts text into numerical features using **TF-IDF vectorization**.
- 🤖 Trains a classification model (**Naive Bayes / Logistic Regression**).
- 📊 Evaluates model with accuracy, precision, recall, and F1-score.
- 📨 Predicts spam/ham for new input emails.

## 📂 Project Structure
```
spamemailclassifier/
│
├── data/
│   └── spam.csv               # Dataset
│
├── spamclassifier.py           # Model training script
├── predict.py                  # Prediction script
├── requirements.txt            # Dependencies
└── README.md                   # Project documentation
```

## 🛠 Installation
1. **Clone the repository**
```bash
git clone https://github.com/vasudha0615/spamemailclassifier.git
cd spamemailclassifier
```

2. **Install dependencies**
```bash
pip install -r requirements.txt
```

## 🚀 Usage
1. **Train the model**
```bash
python spamclassifier.py
```

2. **Predict new email**
```bash
python predict.py
```

## 📊 Dataset
- Dataset used: **[Spam.csv](https://www.kaggle.com/datasets/uciml/sms-spam-collection-dataset)**
- Place `spam.csv` inside the `data/` folder before training.

## 📈 Results
| Metric      | Score   |
|-------------|---------|
| Accuracy    | 97%     |
| Precision   | 96%     |
| Recall      | 98%     |
| F1-score    | 97%     |

## 📜 License
This project is licensed under the **MIT License**.

## 🙌 Acknowledgements
- Dataset: [UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/datasets/sms+spam+collection)
- Libraries: Python, scikit-learn, pandas, numpy, matplotlib, seaborn
