# cunda
CUDA + OpenCV  
* Works only on x64 Windows

### How to run:
1. Install Nvidia CUDA using the tutorial on https://docs.nvidia.com/cuda/cuda-installation-guide-microsoft-windows/index.html (follow the tutorial until step 2.5 including)
2. Clone the repo using `git clone https://github.com/LaBlazer/cunda.git`
3. Download compiled OpenCV libraries from https://mega.nz/#!M9J11K4a!c_ZtDiboLcPJXvQJ1S9r5IH4QuF60FdT2wbTnIe-TlE and extract them to the `lib` folder (overwrite the files)
4. Install Visual Studio CMake tools (Visual Studio Installer->More->Modify->Individual Components->Visual C++ tools for CMake->Modify)
5. Open the project as folder in Visual Studio (File->Open->Folder)
6. Select the correct startup item `cunda.exe` and run
