#!/bin/bash
# Simple script to create a new task template

# Path to virtual environment Python
VENV_PYTHON="/home/h3110fr13nd/Desktop/dev/aiml/infinite-ml/.venv/bin/python"

# Check if Python exists in the virtual environment
if [ ! -f "$VENV_PYTHON" ]; then
    echo "Error: Virtual environment Python not found at $VENV_PYTHON"
    echo "Please make sure the virtual environment is properly set up"
    exit 1
fi

# Run the task generator with the virtual environment's Python
# Add -v flag for verbose output to help debug import issues
echo "Running task generator using virtual environment Python..."
"$VENV_PYTHON" -v -m infinite_ml.tools.task_generator "$@"

# If you continue having issues with circular imports, try this alternative approach
# by using PYTHONPATH to specify the module location
# echo "If the above fails, trying alternative approach..."
# PYTHONPATH="/home/h3110fr13nd/Desktop/dev/aiml/infinite-ml" "$VENV_PYTHON" -v -m infinite_ml.tools.task_generator "$@"
