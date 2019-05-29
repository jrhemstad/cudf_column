// compile with `nvcc --std=c++14 example.cu`

#include "cuda_memory_resource.cuh"
#include "device_memory_resource.cuh"

#include <cstdio>

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
  cudf::mr::device_memory_resource* resource = new cudf::mr::cuda_memory_resource();

  void* data = resource->allocate(size * sizeof(int));

  init<<<1, 256>>>(static_cast<int*>(data));
  print<<<1, 256>>>(static_cast<int*>(data));

  resource->deallocate(data);
}