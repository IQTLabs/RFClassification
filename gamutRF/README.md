# GamutRF C++ Inference
1. To create a C++ enabled version of the GamutRF PyTorch model: 
    ```
    git clone RFClassification 
    cd RFClassification
    python gamutRF/gamutrf_cpp_convert.py
    ```
    This will create the C++ enabled model file `RFClassification/gamutRF/cpp/traced_gamutrf_model.pt`



2. Install LibTorch 
    - Download LibTorch from the [PyTorch website](https://pytorch.org/get-started/locally/)
    - Unzip the downloaded file. Make note of the unzipped directory for use below.



3. To run the C++ enabled version:
    ```
    cd RFClassification/gamutRF/cpp
    mkdir build
    cd build
    cmake -DCMAKE_PREFIX_PATH=/path/to/libtorch ..
    cmake --build . --config Release
    ./gamutrf_inference ../traced_gamutrf_model.pt
    ```
