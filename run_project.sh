#!/bin/bash
# Master Runner & Interactive Demo - Ali Cihan Ozdemir (9091405)

echo "=== PathoIntern Master Runner ==="
echo "Installing requirements..."
pip install -r requirements.txt

echo -e "\n=== Running MLOps Tests ==="
python -m pytest tests/

echo -e "\n=== Launching Interactive Demo ==="
python src/scripts/interactive_demo.py
