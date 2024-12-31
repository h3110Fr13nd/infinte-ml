try:
    from .cuda_utils import get_cuda_device_info, get_device_count
except ImportError:
    # Provide fallback if extension is not built
    import warnings
    warnings.warn("CUDA extension not built or CUDA not available. Some functionality will be limited.")
    
    def get_cuda_device_info():
        return "CUDA extension not available. Please build extension with CMake."
    
    def get_device_count():
        return 0

__all__ = ['get_cuda_device_info', 'get_device_count']
