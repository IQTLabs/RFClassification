#include <torch/script.h> // One-stop header.

#include <iostream>
#include <memory>
#include <chrono>



int main(int argc, const char* argv[]) {
    if (argc != 2) {
        std::cerr << "usage: gamutrf_inference <path-to-exported-script-module>\n";
        return -1;
    }


    torch::jit::script::Module module;
    try {
        // Deserialize the ScriptModule from a file using torch::jit::load().
        module = torch::jit::load(argv[1]);
    }
    catch (const c10::Error& e) {
        std::cerr << "error loading the model\n";
        return -1;
    }

    std::cout << "ok\n";

    // Create a vector of inputs.
    std::vector<torch::jit::IValue> inputs;
    inputs.push_back(torch::ones({1, 3, 256, 256}));

    // Execute the model and turn its output into a tensor.
    std::chrono::steady_clock::time_point begin = std::chrono::steady_clock::now();

    at::Tensor output = module.forward(inputs).toTensor();

    std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();
    
    std::cout << "Time difference (sec) = " <<  (std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count()) /1000000.0  <<std::endl;
    std::cout << "Time difference = " << std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count() << "[µs]" << std::endl;
    std::cout << "Output = " << output << '\n'; 

}