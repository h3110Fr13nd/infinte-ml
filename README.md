# Infinite ML

A project for building and learning machine learning algorithms from scratch with GPU acceleration.

## Project Structure

```
infinite_ml/
├── cmake/                     # CMake modules and helpers
├── docs/                      # Documentation
│   ├── api/                   # API documentation
│   └── templates/             # Templates for new tasks
├── infinite_ml/               # Main Python package
│   ├── cli.py                 # Command line interface
│   ├── common/                # Shared utilities and code
│   │   ├── binding_utils/     # Pybind11 utilities
│   │   ├── cuda/              # Common CUDA utilities
│   │   └── utils/             # Python utilities
│   └── tasks/                 # Machine learning tasks/algorithms
│       ├── 01_vector_addition/  # Example: Vector addition
│       │   ├── cuda/          # CUDA implementation
│       │   ├── numpy/         # NumPy implementation
│       │   ├── pytorch/       # PyTorch implementation
│       │   ├── __init__.py    # Package init
│       │   ├── interface.py   # Unified interface
│       │   └── config.py      # Task configuration
│       └── ...                # More tasks
├── CMakeLists.txt             # Main CMake file
├── setup.py                   # Package setup script
└── README.md                  # This file
```

## Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/infinite-ml.git
cd infinite-ml

# Install dependencies
pip install -e .

# Build CUDA extensions
mkdir build && cd build
cmake ..
make -j
make install
```

## Usage

### Command Line Interface

```bash
# List available tasks
python -m infinite_ml.cli list

# Run a specific task
python -m infinite_ml.cli run 01_vector_addition

# Benchmark a task
python -m infinite_ml.cli benchmark 01_vector_addition
```

### Python API

```python
import numpy as np
from infinite_ml.tasks import get_task

# Get a task implementation
vector_add = get_task("01_vector_addition", "cuda")

# Use the implementation
a = np.random.random(1000).astype(np.float32)
b = np.random.random(1000).astype(np.float32)
result = vector_add(a, b)
```

## Adding a New Task

1. Create a new directory in `infinite_ml/tasks/` with your task name
2. Implement the interface.py pattern shown in the examples
3. Create implementations in the appropriate subdirectories (cuda, numpy, pytorch)
4. Add your task config.py to define default parameters
5. Create a benchmark.py file to use with the benchmarking system

## License

[MIT License](LICENSE)
