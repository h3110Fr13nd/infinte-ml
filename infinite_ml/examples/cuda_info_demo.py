from infinite_ml.common.cuda import get_cuda_device_info, get_device_count

def main():
    print(f"Found {get_device_count()} CUDA devices")
    print("\nCUDA Device Information:")
    print("-----------------------")
    print(get_cuda_device_info())

if __name__ == "__main__":
    main()
