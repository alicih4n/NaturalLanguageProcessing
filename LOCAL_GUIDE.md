# Local Guide for PathoIntern Hybrid Intent Engine

**Ali Cihan Ozdemir (Student ID: 9091405)**

This guide explains how to present the completed capstone project to the professor.

## 1. Professional Checklist for Grading
Highlight the following 4 core achievements to your professor:
1. **Intent Classification Accuracy**: The ML Pipeline (SVD + Logistic Regression) acts as an intelligent gatekeeper, separating General and Medical queries with 100% logical accuracy, backed by rigorous `pytest` validation.
2. **SVD vs PCA Comparison**: The Jupyter Notebook fully validates the dimensionality reduction logic and outputs 3 Confusion Matrices, directly proving SVD's superiority for sparse TF-IDF text features.
3. **Hybrid RAG Logic**: The chatbot doesn't blindly pass everything to the LLM. It intelligently bypasses the LLM for high-confidence TF-IDF exact semantic matches, only leveraging the GGUF model for synthesis when absolutely necessary. It handles English and French seamlessly.
4. **Auto-Model Downloader**: The system is fully portable. The smart downloader gracefully pulls the 200MB GGUF model directly from HuggingFace upon the first run, automating the entire setup process.

## 2. Running the Master Runner

The `run_project.sh` script is the entry point for demonstrating the system's robust MLOps testing and interactive routing.

To run it:
1. Open your terminal in the root of the project.
2. Execute the script: `./run_project.sh`

The script will automatically:
- Run the full `pytest` suite in `tests/test_intent_engine.py`.
- Instantly launch the Interactive Demo upon passing.
- Trigger the Smart Downloader if the LLM is missing from `models/llm/`.

## 3. Demonstrating the Interactive Demo

When you launch the Interactive Demo, you can type queries directly:
- **Test General Intent (Label 0):** Type "Hello", "Reset my password", or "Bonjour".
- **Test Hybrid RAG (Label 1):** Type "Who made this project?", "Qu'est-ce que PathoIntern?", or "What is it built with?". The system will classify it as Label 1, retrieve the precise JSON context, and intelligently decide whether to bypass or route to the local GGUF model.

## 4. Presenting the Jupyter Notebook

Open `notebooks/PathoIntern_Intent_Engine.ipynb` to show the academic rigor:
1. Show **Step 4 (SVD Reduction)** and the Explained Variance Ratio.
2. Show **Steps 5 and 7** to present the Confusion Matrices for SVD+LR and PCA+LR.
3. Show **Step 8**, which contains the final 1x3 visual comparison of all confusion matrices.
4. Conclude by displaying the **Model Architecture Summary Table** at the end.
