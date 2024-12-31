from setuptools import setup, find_packages
from setuptools.extension import Extension
from setuptools.command.build_ext import build_ext
import os
import sys
import subprocess
import glob
import shutil

# Check for CUDA availability
try:
    nvcc_path = subprocess.check_output(['which', 'nvcc']).decode('utf-8').strip()
    cuda_available = True
    print(f"CUDA compiler found at: {nvcc_path}")
except (subprocess.CalledProcessError, FileNotFoundError):
    cuda_available = False
    print("Warning: CUDA compiler not found. CUDA extensions will not be built.")

# Custom build command to handle CMake compilation
class CMakeExtension(Extension):
    def __init__(self, name, sourcedir=''):
        Extension.__init__(self, name, sources=[])
        self.sourcedir = os.path.abspath(sourcedir)

class CMakeBuild(build_ext):
    def run(self):
        # Check for CMake
        try:
            cmake_version = subprocess.check_output(['cmake', '--version']).decode('utf-8')
            print(f"Using CMake: {cmake_version.split()[2]}")
        except (subprocess.CalledProcessError, FileNotFoundError):
            raise RuntimeError("CMake must be installed to build the extensions")
        
        for ext in self.extensions:
            self.build_extension(ext)
    
    def build_extension(self, ext):
        extdir = os.path.abspath(os.path.dirname(self.get_ext_fullpath(ext.name)))
        
        # Get the Python site-packages directory
        import site
        python_site_packages = site.getsitepackages()[0]
        print(f"Python site-packages directory: {python_site_packages}")
        
        # Create build directory
        build_temp = os.path.join(self.build_temp, ext.name)
        os.makedirs(build_temp, exist_ok=True)
        
        # Create lib directory where .so files will be placed during build
        lib_dir = os.path.join(build_temp, "lib")
        os.makedirs(lib_dir, exist_ok=True)
        
        # Enable verbose CMake output for debugging
        os.environ['VERBOSE'] = '1'
        
        # CMake arguments
        cmake_args = [
            f'-DCMAKE_INSTALL_PREFIX={python_site_packages}',
            f'-DPYTHON_SITE_PACKAGES={python_site_packages}',
            f'-DPYTHON_EXECUTABLE={sys.executable}',
            '-DCMAKE_VERBOSE_MAKEFILE=ON'
        ]
        
        # Build type
        cfg = 'Debug' if self.debug else 'Release'
        build_args = ['--config', cfg]
        cmake_args += [f'-DCMAKE_BUILD_TYPE={cfg}']
        
        # Set parallel jobs based on CPU cores
        import multiprocessing
        jobs = multiprocessing.cpu_count()
        
        if os.name == 'nt':  # Windows
            build_args += ['--', f'/m:{jobs}']
        else:  # Unix
            build_args += ['--', f'-j{jobs}']
        
        # Handle CUDA
        if cuda_available:
            cmake_args += ['-DUSE_CUDA=ON']
        else:
            cmake_args += ['-DUSE_CUDA=OFF']
        
        # Run CMake
        print(f"Running CMake with args: {cmake_args}")
        subprocess.check_call(['cmake', ext.sourcedir] + cmake_args, cwd=build_temp)
        
        # Build
        print("Building extension...")
        subprocess.check_call(['cmake', '--build', '.'] + build_args, cwd=build_temp)
        
        # Install
        print(f"Installing to {python_site_packages}")
        subprocess.check_call(['cmake', '--install', '.'], cwd=build_temp)
        
        # Copy any .so files from build directory to the package directories
        print("Copying .so files to package directories...")
        self.copy_extension_files(build_temp, "infinite_ml/common/cuda")
        for task_dir in glob.glob(os.path.join('infinite_ml/tasks/task_*')):
            if os.path.isdir(task_dir):
                cuda_dir = os.path.join(task_dir, 'cuda')
                self.copy_extension_files(build_temp, cuda_dir)
        
        print(f"Extension {ext.name} built successfully")
    
    def copy_extension_files(self, build_dir, target_dir):
        """Copies .so files from build directory to target package directory"""
        # Ensure target directory exists
        os.makedirs(target_dir, exist_ok=True)
        
    
        target_basename = os.path.basename(target_dir)
        target_parent = os.path.basename(os.path.dirname(target_dir))         
        # Create more specific pattern that includes the parent directory structure
        if target_parent == "cuda" and "tasks" not in target_dir:
            # For common/cuda directory
            pattern = os.path.join(build_dir, f"**/infinite_ml/common/cuda/*.so")
            pattern2 = os.path.join(build_dir, f"lib/infinite_ml/common/cuda/*.so")
        elif "tasks" in target_dir:
            # For task specific cuda directories
            task_path = "/".join(target_dir.split("/")[-3:])  # Get task_xxx/cuda path
            pattern = os.path.join(build_dir, f"**/infinite_ml/{task_path}/*.so")
            pattern2 = os.path.join(build_dir, f"lib/infinite_ml/{task_path}/*.so")
        else:
            # Fall back to the previous pattern for any other directories
            pattern = os.path.join(build_dir, f"**/{target_basename}/*.so")
            pattern2 = os.path.join(build_dir, f"lib/{target_basename}/*.so")         
        so_files = glob.glob(pattern, recursive=True) + glob.glob(pattern2, recursive=True)
        
        # Print for debugging
        print(f"Found SO files for {target_dir}: {so_files}")

        for so_file in so_files:
            # Copy the .so file to the target directory
            dest = os.path.join(target_dir, os.path.basename(so_file))
            print(f"Copying {so_file} to {dest}")
            shutil.copy2(so_file, dest)

        # Make sure all shared object files are executable
        for filename in os.listdir(target_dir):
            if filename.endswith('.so'):
                filepath = os.path.join(target_dir, filename)
                st = os.stat(filepath)
                os.chmod(filepath, st.st_mode | 0o111)  # Add executable bit

# Find CMake extensions to build
def generate_extensions():
    extensions = []
    
    # The main extension that builds everything
    extensions.append(CMakeExtension('infinite_ml'))
    
    return extensions

# Read requirements
with open('requirements.txt') as f:
    requirements = f.read().splitlines()

setup(
    name="infinite_ml",
    version="0.1.0",
    author="Your Name",
    author_email="your.email@example.com",
    description="Machine Learning from First Principles",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/infinite-ml",
    packages=find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.6",
    install_requires=requirements,
    ext_modules=generate_extensions(),
    cmdclass={"build_ext": CMakeBuild},
    zip_safe=False,
    include_package_data=True,
)