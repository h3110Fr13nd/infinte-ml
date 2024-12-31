import importlib
import os
import glob

# Dynamically discover all tasks
TASK_REGISTRY = {}

def _discover_tasks():
    task_dirs = [d for d in glob.glob(os.path.join(os.path.dirname(__file__), "*"))
                if os.path.isdir(d) and not os.path.basename(d).startswith("__")]
    
    for task_dir in task_dirs:
        task_name = os.path.basename(task_dir)
        try:
            # Try importing the interface
            interface_module = importlib.import_module(f".{task_name}.interface", package="infinite_ml.tasks")
            TASK_REGISTRY[task_name] = interface_module.get_implementation
        except (ImportError, AttributeError):
            pass  # Skip if no proper interface exists

# Discover tasks when this module is imported
_discover_tasks()

# Function to get a specific task implementation
def get_task(task_name, impl_type="cuda"):
    if task_name not in TASK_REGISTRY:
        raise ValueError(f"Task '{task_name}' not found. Available tasks: {list(TASK_REGISTRY.keys())}")
    
    try:
        return TASK_REGISTRY[task_name](impl_type)
    except Exception as e:
        raise RuntimeError(f"Failed to load implementation '{impl_type}' for task '{task_name}': {str(e)}")