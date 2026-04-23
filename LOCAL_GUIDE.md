# 📋 Evaluator's Quick-Start Guide

Welcome to the **PathoIntern** grading and evaluation guide. This document is specifically designed to help professors and evaluators quickly locate all components necessary to assess the project against the grading rubrics for **PROG8245** and **CSCN8010**.

---

## 🎯 Grading Rubric Roadmap

To streamline your review process, we have mapped out exactly where each core requirement has been implemented.

### 1. Classical Machine Learning Requirements
**Rubric Targets:** *TF-IDF, SVD, PCA, Naive Bayes, Confusion Matrices*
* **Location:** [`notebooks/PathoIntern_Intent_Engine.ipynb`](notebooks/PathoIntern_Intent_Engine.ipynb)
* **Description:** This extensive Jupyter Notebook contains the end-to-end data pipeline. It features the construction of the classical ML `Fast-Intent-Router`, including data preprocessing, feature extraction via **TF-IDF**, dimensionality reduction using **PCA** and **SVD**, and the training and evaluation of models like **Naive Bayes** and Logistic Regression. All visualizations, including **Confusion Matrices** and ROC curves, are generated here.

### 2. Robustness & Automated Testing
**Rubric Targets:** *Unit Tests, Pipeline Validation*
* **Location:** [`tests/`](tests/)
* **Description:** This directory houses the pytest suite. You can evaluate our automated validation logic, designed to ensure our intent classification and routing mechanisms function properly without regression. Run the tests using `python -m pytest tests/`.

### 3. Application & Real-World Integration
**Rubric Targets:** *Working Prototype, Eager Loading, Hybrid RAG Implementation*
* **Location:** [`src/scripts/interactive_demo.py`](src/scripts/interactive_demo.py)
* **Description:** This is the entry point for the **Live Chatbot Engine**. It integrates the classical classification pipeline with our deeply quantized GGUF LLM RAG setup. It demonstrates eager loading, the automated fallback LLM downloader (fetching the ~200MB model automatically), and bilingual (French/English) retrieval.

---

## 🚀 Quick Run Instructions

For an immediate demonstration of the system:

1. **Verify the Environment:** Ensure your virtual environment is active and `requirements.txt` dependencies are installed.
2. **Run Tests:** `python -m pytest tests/`
3. **Start Chatbot:** `python src/scripts/interactive_demo.py`

*(For detailed OS-specific setup, please refer to the comprehensive [README.md](README.md).)*

Thank you for reviewing **PathoIntern**!
