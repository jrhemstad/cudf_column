#pragma once

#include "device_memory_resource.hpp"

#include <cuda_runtime_api.h>
#include <cassert>
#include <exception>
#include <iostream>

namespace rmm {
namespace mr {
/**---------------------------------------------------------------------------*
 * @brief `device_memory_resource` derived class that uses cudaMalloc/Free for
 * allocation/deallocation.
 *---------------------------------------------------------------------------**/
class cuda_memory_resource final : public device_memory_resource {
  bool supports_streams() const noexcept override { return false; }

 private:
  /**---------------------------------------------------------------------------*
   * @brief Allocates memory of size at least \p bytes using cudaMalloc.
   *
   * The returned pointer has at least 256B alignment.
   *
   * @param bytes The size of the allocation
   * @return void* Pointer to the newly allocated memory
   *---------------------------------------------------------------------------**/
  void* do_allocate(std::size_t bytes, cudaStream_t) override {
    void* p{nullptr};
    cudaError_t status = cudaMalloc(&p, bytes);
    if (cudaSuccess != status) {
      std::cerr << "cudaMalloc failed: " << cudaGetErrorName(status) << " "
                << cudaGetErrorString(status) << "\n";
      throw std::bad_alloc{};
    }
    return p;
  }

  /**---------------------------------------------------------------------------*
   * @brief Deallocate memory pointed to by \p p.
   *
   * If supported, this operation may optionally be executed on a stream.
   * Otherwise, the stream is ignored and the null stream is used.
   *
   * @param p Pointer to be deallocated
   * @param stream Stream on which to perform deallocation
   *---------------------------------------------------------------------------**/
  void do_deallocate(void* p, std::size_t, cudaStream_t) override {
    assert(cudaSuccess == cudaFree(p));
  }
};

}  // namespace mr
}  // namespace rmm
