import sys
import platform
import os

def get_system_info():
    """Gather system information to help with debugging."""
    info = {
        "platform": platform.platform(),
        "python_version": sys.version,
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES", "Not set"),
        "pwd": os.getcwd(),
        "path": sys.path
    }
    
    # Try to detect CUDA through nvidia-smi if available
    try:
        import subprocess
        result = subprocess.run(['nvidia-smi'], stdout=subprocess.PIPE)
        info["nvidia_smi"] = result.stdout.decode('utf-8')
    except:
        info["nvidia_smi"] = "Not available"
        
    return info

try:
    from infinite_ml.tasks.task_001_addition.cuda import addition
    result = addition.addition()
    
    if result.success:
        print(f"CUDA Success: {result.message}")
    else:
        print(f"CUDA Error: {result.error_message}")
        sys_info = get_system_info()
        print("\nSystem information:")
        for key, value in sys_info.items():
            print(f"  {key}: {value}")
except Exception as e:
    print(f"Failed to import or run CUDA module: {e}")
    import traceback
    traceback.print_exc()

if __name__ == "__main__":
    pass  # Already executed above
