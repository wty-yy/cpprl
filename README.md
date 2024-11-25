`settings.json`
```json
{
  "cmake.configureArgs": [
    "-DCMAKE_PREFIX_PATH=/home/yy/lib/libtorch-2.5.1+cu121",
    "-DCMAKE_CUDA_COMPILER=/usr/local/cuda/bin/nvcc",
    "-DCAFFE2_USE_CUDNN=True"
  ]
}
```

`c_cpp_properties.json`
```json
{
    "configurations": [
        {
            "name": "Linux",
            "includePath": [
                "${workspaceFolder}/**",
                "/home/yy/lib/libtorch-2.5.1+cu121/include/torch/csrc/api/include",
                "/home/yy/lib/libtorch-2.5.1+cu121/include",
                "/usr/local/include/opencv4"
            ],
            "defines": [],
            "compilerPath": "/usr/bin/gcc",
            "intelliSenseMode": "linux-gcc-x64",
            "configurationProvider": "ms-vscode.cmake-tools"
        }
    ],
    "version": 4
}
```
