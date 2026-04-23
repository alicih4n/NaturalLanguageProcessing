# PathoIntern: Multilingual Intent Classification

## Project Overview

This project implements **Phase 1 and Phase 2** of the PathoIntern Data Science initiative. We introduce a newly augmented dataset and a Jupyter Notebook that establishes a baseline classical machine learning model for intent classification.

## Hybrid Architecture Vision

The ML model created in this repository acts as a **"Fast-Intent-Router"** for the existing GGUF model (`LFM2.5-350M-Q4_K_M.gguf`). 

Instead of routing every single user query directly to the computationally expensive LLM, our classical Machine Learning pipeline (Multinomial Naive Bayes trained on TF-IDF features) intercepts queries first. 

- **Label 1 (Medical/Pathology):** If the router detects a medical or pathology-related query, it will forward it to the specialized GGUF model to generate an accurate, domain-specific response using the `chatbot_engine.py`.
- **Label 0 (General/System):** If the query is identified as a general greeting or system command (e.g., "Hello", "Reset my password"), the router bypasses the LLM and handles it via lightweight predefined logic.

This hybrid approach guarantees fast response times for general interactions while reserving the heavy lifting for the GGUF model when domain expertise is actually required.

## Components

1. **Synthetic Dataset Generation:** 
   The script `generate_dataset.py` takes our foundational 155 JSON QA pairs (English & French) and programmatically augments them into a robust 2,000-row CSV dataset (`data/pathology_classification_dataset.csv`).
2. **Jupyter Notebook Implementation:**
   The notebook `PathoIntern_Intent_Engine.ipynb` demonstrates the ingestion, TF-IDF vectorization, and dimensionality reduction (SVD vs PCA) of the dataset. The Intent Classification engine is now complete and validated with 3 different architectures (Multinomial Naive Bayes, SVD + Logistic Regression, PCA + Logistic Regression), successfully acting as the proof-of-concept for the Fast-Intent-Router.
3. **Hybrid Bridge Demo:**
   The script `src/predict_intent.py` acts as a working demonstration, correctly identifying and routing interactions.

## Project Structure

- `data/raw/`: Base JSON knowledge bases.
- `data/processed/`: Augmented classification datasets.
- `models/classifiers/`: Stores serialized `joblib` artifacts (TF-IDF vectorizer, SVD, LR).
- `models/llm/`: Domain-specific GGUF base models.
- `notebooks/`: The core academic pipeline validating TF-IDF, SVD, and PCA implementations.
- `src/core/`: Application backend (`chatbot_engine.py`).
- `src/scripts/`: Generator scripts and the `interactive_demo.py` hybrid bridge.
- `tests/`: Robust MLOps `pytest` suite for inference consistency.

## How to Run

1. Ensure Python 3.10+ is installed.
2. Run the automated master script, which handles dependencies, testing, and model downloading:
   ```bash
   ./run_project.sh
   ```

## Model Acquisition

The system features a Smart Downloader. On the first run, the interactive demo will automatically check if the local LLM exists. If missing, it will prompt you to download the quantized 350M GGUF model directly from HuggingFace to enable the Hybrid RAG capabilities. If skipped, the chatbot will seamlessly fallback to "Lightweight Mode" using only JSON semantic retrieval.

## Contributor Info

**Ali Cihan Ozdemir** (Student ID: 9091405)
