// compile with `nvcc --std=c++14 example.cu`

#include <cassert>
#include <cstdio>
#include "default_memory_resource.hpp"
#include "device_buffer.hpp"

constexpr int size{10};

__global__ void init(int* data) {
  if (threadIdx.x < size) {
    data[threadIdx.x] = threadIdx.x;
  }
}

__global__ void print(int* data) {
  if (threadIdx.x < size) {
    printf("%d\n", data[threadIdx.x]);
  }
}

int main(void) {
  std::size_t size_in_bytes = size * sizeof(int);
  rmm::device_buffer buff(size_in_bytes);

  init<<<1, 256>>>(static_cast<int*>(buff.data()));
  print<<<1, 256>>>(static_cast<int*>(buff.data()));

  assert(cudaSuccess == cudaDeviceSynchronize());
}