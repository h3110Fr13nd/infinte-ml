import numpy as np
import infinite_ml.tasks.task_001_addition.cuda.addition as ca
import time

def main():
    # Initialize logger
    ca.initialize_logger("python_test.log")
    
    # Test different sizes
    sizes = [1024, 2048, 4096, 8192, 16384, 32768, 65536, 131072, 262144, 524288, 1048576, 2097152, 4194304, 8388608, 16777216, 33554432, 67108864]
    threads_per_block_options = [32, 64, 128, 256, 512, 1024]
    for i in range(10):
        for size in sizes:
            for threads_per_block in threads_per_block_options:
                print(f"Testing size {size} with {threads_per_block} threads per block")
                
                # Create numpy arrays
                a = np.random.rand(size).astype(np.float32)
                b = np.random.rand(size).astype(np.float32)
                c_cpu = np.zeros_like(a)
                c_cuda = np.zeros_like(a)
                
                # CPU calculation
                cpu_result = ca.vector_add_cpu(a, b, c_cpu)
                print("CPU result:")
                print(cpu_result)
                ca.log_performance_results_cpu(cpu_result)
                
                # CUDA calculation
                cuda_result = ca.vector_add_cuda(a, b, c_cuda, threads_per_block)
                print("CUDA result:")
                print(cuda_result)
                ca.log_performance_results_cuda(cuda_result)
                
                # Verify the results
                ca.verify_result(a, b, c_cuda)
                
                # Also verify using numpy
                c_numpy = a + b
                np.testing.assert_allclose(c_cpu, c_numpy, rtol=1e-5)
                np.testing.assert_allclose(c_cuda, c_numpy, rtol=1e-5)
                
                print("All verifications passed!")
                print("=" * 50)
    
    # Clean up logger
    ca.cleanup_logger()

if __name__ == "__main__":
    main()
