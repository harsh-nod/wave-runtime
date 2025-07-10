# Wave Runtime

A Python extension module built with nanobind that provides a performant runtime for the Wave language. This runtime serves as a wrapper over `hipModuleLaunchKernel` to enable efficient GPU kernel execution on AMD hardware using HIP (AMD ROCm).

## Overview

Wave Runtime is the execution engine for the Wave language that provides:
- **Kernel Launch Interface**: Direct wrapper over `hipLaunchKernel` for GPU execution
- **Binary Loading**: Dynamic loading of HIP kernel binaries
- **Memory Management**: Efficient tensor and scalar argument handling
- **nanobind**: For efficient Python-C++ bindings
- **HIP Integration**: AMD GPU acceleration via ROCm

## Prerequisites

### System Requirements
- **Linux** (Ubuntu 20.04+ recommended)
- **AMD GPU** with ROCm support
- **Python 3.10+**

### Required Dependencies

1. **Python Dependencies**
   ```bash
   pip install nanobind
   ```

2. **System Dependencies**
   ```bash
   # Ubuntu/Debian
   sudo apt update
   sudo apt install cmake build-essential python3-dev

   # Install ROCm (AMD GPU support)
   # Follow instructions at: https://rocmdocs.amd.com/en/latest/Installation_Guide/Installation-Guide.html
   ```

3. **ROCm Installation**
   ```bash
   # Add ROCm repository
   sudo mkdir --parents --mode=0755 /etc/apt/keyrings
   wget https://repo.radeon.com/rocm/rocm.gpg.key -O - | \
       sudo gpg --dearmor | sudo tee /etc/apt/keyrings/rocm-keyring.gpg > /dev/null

   echo 'deb [arch=amd64 signed-by=/etc/apt/keyrings/rocm-keyring.gpg] https://repo.radeon.com/rocm/apt/debian jammy main' | \
       sudo tee /etc/apt/sources.list.d/rocm.list

   # Install ROCm
   sudo apt update
   sudo apt install rocm-hip-sdk
   ```

## Building

### Quick Build
```bash
# Clone the repository
git clone <repository-url>
cd wave_runtime

cmake -B build -S . -DCMAKE_LIBRARY_OUTPUT_DIRECTORY=.
cmake --build build
```

### Build Options

1. **Debug Build**
   ```bash
   cmake -DCMAKE_BUILD_TYPE=Debug ..
   make -j$(nproc)
   ```

2. **Release Build** (default)
   ```bash
   cmake -DCMAKE_BUILD_TYPE=Release ..
   make -j$(nproc)
   ```

3. **Install to System**
   ```bash
   cmake ..
   make -j$(nproc)
   sudo make install
   ```

## Usage

### Basic Example


```python
import wave_runtime

# Load a HIP kernel binary
module_capsule, function_ptr = wave_runtime.load_binary("kernel.bin", "my_kernel")

# Create kernel launch configuration
launch_info = wave_runtime.KernelLaunchInfo(
    gpu_func=function_ptr,
    sharedMemoryBytes=0,
    gridX=1, gridY=1, gridZ=1,
    blockX=256, blockY=1, blockZ=1
)

# Launch kernel with tensor arguments and scalar parameters
tensors = [tensor1_ptr, tensor2_ptr, output_ptr]  # uint64_t pointers
dynamic_dims = [dim1, dim2]  # Dynamic dimensions
scalar_args = [42, 3.14]  # Scalar arguments

wave_runtime.launch(launch_info, tensors, dynamic_dims, scalar_args)
```

## Project Structure

```
wave_runtime/
├── CMakeLists.txt      # Build configuration
├── runtime.cpp         # Main C++ implementation with hipLaunchKernel wrapper
├── setup.py           # Python package setup
├── test.py            # Example usage
└── README.md          # This file
```

## Core Components

### `runtime.cpp`
The main implementation file that provides:
- **`launch()`**: Wrapper function over `hipLaunchKernel` for executing GPU kernels
- **`load_binary()`**: Dynamic loading of HIP kernel binaries from disk
- **`KernelLaunchInfo`**: Configuration structure for kernel launch parameters
- **Memory Management**: Efficient handling of tensor pointers, dynamic dimensions, and scalar arguments
- **Error Handling**: Comprehensive HIP error checking and reporting

## Troubleshooting

### Common Issues

1. **nanobind not found**
   ```bash
   pip install nanobind
   ```

2. **HIP not found**
   - Ensure ROCm is properly installed
   - Check that `/opt/rocm` exists and contains HIP libraries
   - Verify GPU drivers are installed

3. **Kernel binary loading fails**
   - Ensure the HIP kernel binary exists and is accessible
   - Verify the kernel function name matches the binary
   - Check that the binary was compiled for the correct GPU architecture

4. **Kernel launch fails**
   - Verify tensor pointers are valid GPU memory addresses
   - Check that grid and block dimensions are appropriate for your kernel
   - Ensure shared memory requirements don't exceed GPU limits

5. **Build fails on non-Linux platforms**
   - This project only supports Linux with AMD GPUs
   - Windows and macOS builds are not supported

### Verification Commands

```bash
# Check Python dependencies
python -c "import nanobind; print('nanobind OK')"

# Check ROCm installation
ls /opt/rocm/lib/cmake/hip

# Test basic runtime functionality
python -c "import wave_runtime; print('Wave Runtime OK')"
```

## Development

### Adding New Operations

1. **C++ Implementation**: Add functions to `runtime.cpp`
2. **Python Bindings**: Use nanobind to expose functions to Python
3. **Testing**: Add test cases to `test.py`

### Build System

The project uses CMake with the following key features:
- **Platform Detection**: Automatically detects Linux/Windows/macOS
- **Dependency Checking**: Validates all required dependencies
- **Cross-Platform Support**: Currently Linux-only, extensible for other platforms
- **Optimization**: Includes nanobind optimizations for performance

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests
5. Submit a pull request

## Support

For issues and questions:
- Check the troubleshooting section above
- Verify your system meets the prerequisites
- Ensure all dependencies are properly installed
