# End-to-End-ML-pipeline-using-DVC-Aws

An end-to-end **Machine Learning pipeline** for **Spam SMS Classification** built with **DVC (Data Version Control)**, **Git**, and **AWS S3** for scalable and reproducible workflows.  

The project demonstrates how to go from raw data ➝ preprocessing ➝ feature engineering ➝ model training ➝ evaluation while ensuring **automation, version control, and cloud storage** of artifacts.  

---

## 📖 Table of Contents  
- [Overview](#-overview)  
- [Features](#-features)  
- [Tech Stack](#-tech-stack)  
- [Project Structure](#-project-structure)  
- [Workflow](#-workflow)  
- [Installation & Setup](#-installation--setup)  
- [Usage](#-usage)  
- [Results](#-results)  
- [Future Improvements](#-future-improvements)  
- [Acknowledgements](#-acknowledgements)  

---

## 🔎 Overview  
The task is to classify SMS messages as **Spam** or **Ham (Not Spam)** using **Natural Language Processing (NLP)** techniques.  

Key aspects of this project:  
- Modularized ML workflow split into reusable Python scripts.  
- Data & model versioning handled by **DVC**.  
- Storage of large files and artifacts in **AWS S3**.  
- Reproducible and automated experiments with `dvc repro`.  

---

## ✨ Features  
✔️ Data exploration and baseline experiments in Jupyter Notebook  
✔️ Modular pipeline with distinct stages (ingestion → preprocessing → feature engineering → model training → evaluation)  
✔️ NLP processing: text cleaning, **Porter Stemming**, **TF-IDF vectorization**  
✔️ Automated pipeline execution using **DVC**  
✔️ Remote storage of data/models in **AWS S3**  
✔️ Easy reproducibility and collaboration  

---

## ⚙️ Tech Stack  
- **Programming Language**: Python 3  
- **Tools & Platforms**: DVC, Git, AWS S3, Jupyter Notebook  
- **Libraries**:  
  - Data Handling → Pandas, NumPy  
  - NLP → NLTK (Porter Stemmer), Scikit-learn (TfidfVectorizer)  
  - Modeling → Scikit-learn (Logistic Regression, Naive Bayes, etc.)  
  - Evaluation → Accuracy, Precision, Recall, F1-Score  

---

## 📂 Project Structure  

```bash
End-to-End-ML-pipeline-using-DVC-Aws/
│
├── .dvc/                         # DVC internal files
├── .dvclive/                      
├── experiments/                  # Experimental work
│   ├── mynotebook.ipynb          # EDA and prototyping
│   └── spam.csv                  # Original dataset
│
├── src/                          # Modular pipeline source code
│   ├── data_ingestion.py         # Load and validate data
│   ├── data_preprocessing.py     # Text cleaning & stemming
│   ├── feature_Engineering.py    # TF-IDF vectorization
│   ├── model_building.py         # Model training
│   └── model_evaluation.py       # Performance metrics
│
├── params.yaml                   # Configuration parameters
├── dvc.yaml                      # Pipeline definition
├── dvc.lock                      
├── .gitignore                    # Git ignore rules
└── README.md                     # Project documentation
├── .dvcignore
├── .gitignore                    

```

## 🔄 Workflow  

1. **Data Exploration**  
   - Perform EDA & baseline experiments in `experiments/mynotebook.ipynb`.  

2. **Pipeline Stages** (src folder)  
   - `data_ingestion.py` → Load raw dataset.  
   - `data_preprocessing.py` → Clean text (lowercasing, punctuation removal, etc.).  
   - `feature_engineering.py` → Apply stemming & TF-IDF vectorization.  
   - `model_building.py` → Train spam classifier models.  
   - `model_evaluation.py` → Evaluate with metrics like accuracy, F1-score.  

3. **Version Control with DVC**  
   - Data & model files tracked using **DVC**.  
   - Pipeline defined in `dvc.yaml`.  

4. **Cloud Storage with AWS S3**  
   - Configured S3 bucket for storing large data/artifacts.  
   - Artifacts pushed with `dvc push`.  

5. **Reproducibility**  
   - Any contributor can run:  
     ```bash
     dvc repro
     ```
     to reproduce the entire pipeline.  

---

