#pragma once

#include "device_memory_resource.cuh"

#include <cuda_runtime_api.h>

namespace cudf {
namespace mr {
class cuda_memory_resource final : public device_memory_resource {
  bool supports_streams() const noexcept override { return false; }

 private:
  /**---------------------------------------------------------------------------*
   * @brief Allocates memory of size at least \p bytes using cudaMalloc.
   *
   * @param bytes The size of the allocation
   * @return void* Pointer to the newly allocated memory
   *---------------------------------------------------------------------------**/
  virtual void* do_allocate(std::size_t bytes, cudaStream_t) override {
    void* p{nullptr};
    cudaMalloc(&p, bytes);
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
  virtual void do_deallocate(void* p, cudaStream_t) override { cudaFree(p); }
};

}  // namespace mr
}  // namespace cudf
