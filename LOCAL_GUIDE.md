# Local Guide for PathoIntern Hybrid Intent Engine

**Ali Cihan Ozdemir (Student ID: 9091405)**

This guide explains how to present the completed Phase 6 project to the professor.

## 1. Running the Master Runner

The `run_project.sh` script is the entry point for demonstrating the system's robust MLOps testing and interactive routing.

To run it:
1. Open your terminal in the root of the project.
2. Ensure the script is executable: `chmod +x run_project.sh`
3. Execute the script: `./run_project.sh`

The script will automatically:
- Verify all dependencies are installed from `requirements.txt`.
- Run the full `pytest` suite in `tests/test_intent_engine.py` to prove Model Loading, Inference Consistency, Multilingual Support, and Input Safety.
- Ask if you want to launch the Interactive Demo.

## 2. Demonstrating the Interactive Demo

When you launch the Interactive Demo, you can type queries directly:
- **Test General Intent (Label 0):** Type "Hello", "Reset my password", or "Bonjour". The system will indicate it is handling the query with a lightweight standard response.
- **Test Medical Intent (Label 1):** Type "What are blast cells?", "Pathology of blood smear", or "Malaria symptoms". The system will classify it as Label 1 and output that it's routing the request to the local GGUF model.

## 3. Presenting the Jupyter Notebook

Open `notebooks/PathoIntern_Intent_Engine.ipynb` to show the academic rigor:
1. Show **Step 4 (SVD Reduction)** and the Explained Variance Ratio.
2. Show **Steps 5 and 7** to present the Confusion Matrices for SVD+LR and PCA+LR.
3. Show **Step 8**, which contains the final 1x3 visual comparison of all confusion matrices.
4. Conclude by displaying the **Model Architecture Summary Table** at the end.
