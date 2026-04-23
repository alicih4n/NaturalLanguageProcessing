#!/bin/bash
# Master Runner & Interactive Demo - Ali Cihan Ozdemir (9091405)

echo "=== PathoIntern Master Runner ==="
echo "Installing requirements..."
pip install -r requirements.txt

echo -e "\n=== Running MLOps Tests ==="
python -m pytest tests/

echo -e "\n=== Do you want to launch the interactive demo? (y/n) ==="
read launch_demo

if [ "$launch_demo" = "y" ] || [ "$launch_demo" = "Y" ]; then
    python src/scripts/interactive_demo.py
else
    echo "Exiting Master Runner."
fi
