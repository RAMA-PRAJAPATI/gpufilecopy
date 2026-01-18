#include <mpi.h>
#include <cuda_runtime.h>
#include <iostream>
#include <fstream>
#include <vector>
#include <algorithm>

__global__ void gpu_copy(char* data, size_t n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) data[i] = data[i];
}

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    if (argc != 3) {
        if (rank == 0)
            std::cerr << "Usage: filecopy.exe input output\n";
        MPI_Finalize();
        return 1;
    }

    const char* in = argv[1];
    const char* out = argv[2];

    std::ifstream fin(in, std::ios::binary | std::ios::ate);
    if (!fin) {
        if (rank == 0) std::cerr << "Cannot open input file\n";
        MPI_Finalize();
        return 1;
    }

    size_t file_size = fin.tellg();
    fin.seekg(0);

    size_t chunk = (file_size + size - 1) / size;
    size_t start = rank * chunk;
    size_t end = std::min(start + chunk, file_size);
    size_t local_size = end > start ? end - start : 0;

    std::vector<char> buffer(local_size);

    fin.seekg(start);
    fin.read(buffer.data(), local_size);
    fin.close();

    char* d;
    cudaMalloc(&d, local_size);
    cudaMemcpy(d, buffer.data(), local_size, cudaMemcpyHostToDevice);

    int threads = 256;
    int blocks = (local_size + threads - 1) / threads;
    gpu_copy<<<blocks, threads>>>(d, local_size);
    cudaDeviceSynchronize();

    cudaMemcpy(buffer.data(), d, local_size, cudaMemcpyDeviceToHost);
    cudaFree(d);

    if (rank == 0) {
        std::ofstream f(out, std::ios::binary | std::ios::trunc);
        f.seekp(file_size - 1);
        f.write("", 1);
        f.close();
    }

    MPI_Barrier(MPI_COMM_WORLD);

    std::fstream fout(out, std::ios::binary | std::ios::in | std::ios::out);
    fout.seekp(start);
    fout.write(buffer.data(), local_size);
    fout.close();

    if (rank == 0)
        std::cout << "File copied successfully\n";

    MPI_Finalize();
    return 0;
}
