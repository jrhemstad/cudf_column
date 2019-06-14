// compile with `nvcc --std=c++14 example.cu`

#include <cstdio>
#include "default_memory_resource.hpp"

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
  rmm::mr::device_memory_resource* resource = rmm::mr::get_default_resource();
  std::size_t size_in_bytes = size * sizeof(int);

  void* data = resource->allocate(size_in_bytes);

  init<<<1, 256>>>(static_cast<int*>(data));
  print<<<1, 256>>>(static_cast<int*>(data));

  resource->deallocate(data, size_in_bytes);
}