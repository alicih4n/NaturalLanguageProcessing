# 🧬 PathoIntern: Hybrid Intent Classification & RAG Architecture

> **Context:** This repository serves as the final submission for **TWO** distinct courses: **PROG8245** and **CSCN8010**.

---

## 👥 Team & Contributors

### Course 1: PROG8245 (Machine Learning)
* **Ali Cihan Ozdemir** (9091405) - *Lead Contributor & Architect*
* **Lohith Reddy Danda** (9054470)
* **Muthuraj Jayakumar** (9084570)
* **Sumanth Reddy K** (9040660)

### Course 2: CSCN8010 (Foundations of Machine Learning Frameworks)
* **Ali Cihan Ozdemir** (9091405) - *Lead Contributor & Architect*
* **Lohith Reddy Danda** (9054470)
* **Muthuraj Jayakumar** (9084570)

*(Note: Sumanth Reddy K is explicitly excluded from the CSCN8010 submission).*

---

## 🏗️ Architecture Overview

The **PathoIntern** project integrates a classical Machine Learning router with a Large Language Model (LLM) Retrieval-Augmented Generation (RAG) system.

### Fast-Intent-Router
A robust classical ML layer designed to intercept user queries and classify intent:
* **Feature Extraction:** Utilizes **TF-IDF** for vectorization.
* **Dimensionality Reduction:** Employs **SVD** and **PCA** to compress the feature space while retaining variance, ensuring high-speed inference.
* **Classification Engine:** Integrates models like **Logistic Regression** and **Naive Bayes** to achieve highly deterministic, low-latency intent classification.

### GGUF LLM RAG Pipeline
An advanced Retrieval-Augmented Generation system running on a deeply quantized LLM:
* **Eager Loading:** Pre-loads model weights and embeddings into memory for rapid initialization and minimal response times.
* **Bilingual Retrieval:** Features a highly specialized retrieval engine capable of context-aware routing for both **English** and **French** knowledge bases, keeping language paths safely isolated.

---

## ⚙️ Comprehensive Installation Guide (OS-Agnostic)

PathoIntern is designed to run locally. Follow the step-by-step instructions below tailored to your operating system.

### Clone the repository
```bash
git clone <repository-url>
cd NaturalLanguageProcessing
```

### Virtual Environment Setup

**macOS/Linux:**
```bash
python3 -m venv venv
source venv/bin/activate
```

**Windows (CMD/PowerShell):**
```cmd
python -m venv venv
venv\Scripts\activate
```

### Install Dependencies
```bash
pip install -r requirements.txt
```

---

## 🚀 Execution & Usage

PathoIntern features a smart **Auto-Downloader** that automatically fetches an optimized **200MB GGUF model** upon the very first run.

### Option A: Master Script (macOS/Linux)
We provide a convenient shell script to run the project effortlessly:
```bash
chmod +x run_project.sh
./run_project.sh
```

### Option B: Run Manually (All OS / Windows)
If you prefer running the components step-by-step or are on a Windows machine:

1. **Run Automated Tests:**
   ```bash
   python -m pytest tests/
   ```
2. **Launch the Live Chatbot Engine:**
   ```bash
   python src/scripts/interactive_demo.py
   ```

---

## 📂 Directory Structure

```text
NaturalLanguageProcessing/
├── README.md                           # Comprehensive project overview
├── LOCAL_GUIDE.md                      # Evaluator's quick-start and grading guide
├── requirements.txt                    # Project dependencies
├── run_project.sh                      # Master execution script (macOS/Linux)
├── data/                               # Dataset files
├── models/                             # ML models and downloaded GGUF weights
├── notebooks/                          # Jupyter Notebooks and utilities
├── src/                                # Core logic and execution scripts
└── tests/                              # Pytest test suites
```
