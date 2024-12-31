import argparse
from infinite_ml.tasks import TASK_REGISTRY, get_task
import importlib

def list_implementations(task_name):
    """List available implementations for a given task"""
    try:
        # Import the config module to see available implementations
        config = importlib.import_module(f"infinite_ml.tasks.{task_name}.config")
        return getattr(config, "AVAILABLE_IMPLEMENTATIONS", ["cuda", "numpy"])
    except (ImportError, AttributeError):
        # Default implementations if not specified
        return ["cuda", "numpy"]

def main():
    parser = argparse.ArgumentParser(description="Infinite ML toolkit")
    
    # Create subparsers for different commands
    subparsers = parser.add_subparsers(dest="command", help="Command to execute")
    
    # List command
    list_parser = subparsers.add_parser("list", help="List available tasks")
    
    # Run command
    run_parser = subparsers.add_parser("run", help="Run a specific task")
    run_parser.add_argument("task", choices=list(TASK_REGISTRY.keys()), help="Task to run")
    run_parser.add_argument("--impl", default="cuda", help="Implementation to use")
    run_parser.add_argument("--args", nargs="*", help="Arguments to pass to the task")
    
    # Benchmark command
    benchmark_parser = subparsers.add_parser("benchmark", help="Benchmark a specific task")
    benchmark_parser.add_argument("task", choices=list(TASK_REGISTRY.keys()), help="Task to benchmark")
    benchmark_parser.add_argument("--impls", nargs="*", default=["cuda", "numpy"], 
                                help="Implementations to benchmark (default: cuda, numpy)")
    benchmark_parser.add_argument("--sizes", nargs="*", type=int, help="Input sizes to benchmark")
    
    # New task creation command
    create_parser = subparsers.add_parser("create-task", help="Create a new task template")
    create_parser.add_argument("task_id", help='Task ID (e.g., "01_vector_addition")')
    create_parser.add_argument("task_name", help='Human-readable task name (e.g., "Vector Addition")')
    create_parser.add_argument("--function-name", help='Function name for the task (default derived from task_name)')
    create_parser.add_argument("--impls", nargs='+', default=['cuda', 'numpy', 'pytorch'],
                              help='Implementations to generate (default: cuda numpy pytorch)')
    
    args = parser.parse_args()
    
    if args.command == "list":
        print("Available tasks:")
        for task_name in sorted(TASK_REGISTRY.keys()):
            impls = list_implementations(task_name)
            print(f"  - {task_name} (implementations: {', '.join(impls)})")
    
    elif args.command == "run":
        try:
            func = get_task(args.task, args.impl)
            
            # Parse additional arguments if provided
            kwargs = {}
            if args.args:
                for arg in args.args:
                    if '=' in arg:
                        key, value = arg.split('=', 1)
                        # Try to convert to appropriate type
                        try:
                            value = eval(value)
                        except:
                            pass
                        kwargs[key] = value
            
            # Call the function
            result = func(**kwargs)
            print(f"Result: {result}")
        except Exception as e:
            print(f"Error: {str(e)}")
    
    elif args.command == "benchmark":
        from infinite_ml.common.utils.benchmark_utils import benchmark_task
        try:
            results = benchmark_task(args.task, impls=args.impls, sizes=args.sizes)
            
            # Print benchmark results
            print(f"Benchmark results for task '{args.task}':")
            for impl, data in results.items():
                print(f"\nImplementation: {impl}")
                for size, stats in data.items():
                    print(f"  Size {size}:")
                    print(f"    Mean: {stats['mean']:.6f}s")
                    print(f"    Std:  {stats['std']:.6f}s")
                    print(f"    Min:  {stats['min']:.6f}s")
                    print(f"    Max:  {stats['max']:.6f}s")
        except Exception as e:
            print(f"Benchmark error: {str(e)}")
    
    elif args.command == "create-task":
        # Import task generator and create a new task
        from infinite_ml.tools.task_generator import create_task_folder
        create_task_folder(
            args.task_id,
            args.task_name,
            args.function_name,
            impls=args.impls
        )

if __name__ == "__main__":
    main()
