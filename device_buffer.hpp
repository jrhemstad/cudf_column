#pragma once

#include "default_memory_resource.hpp"
#include "device_memory_resource.hpp"

#include <cuda_runtime_api.h>
#include <cassert>

namespace rmm {

/**---------------------------------------------------------------------------*
 * @file device_buffer.hpp
 * @brief RAII construct for device memory allocation
 *---------------------------------------------------------------------------**/

class device_buffer {
 public:
  /**---------------------------------------------------------------------------*
   * @brief Constructs a new device buffer of `size` bytes
   *
   * @param size Size in bytes to allocate in device memory
   * @param stream CUDA stream on which memory may be allocated if the memory
   * resource supports streams, else the null stream is used.
   * @param mr Memory resource to use for the device memory allocation
   *---------------------------------------------------------------------------**/
  device_buffer(std::size_t size, cudaStream_t stream = 0,
                mr::device_memory_resource* mr = mr::get_default_resource())
      : _size{size}, _stream{stream}, _mr{mr} {
    _data = _mr->allocate(size, stream);
  }

  /**---------------------------------------------------------------------------*
   * @brief Copy constructor
   *
   * @param other
   *---------------------------------------------------------------------------**/
  device_buffer(device_buffer const& other)
      : _size{other._size}, _stream{other._stream}, _mr{other._mr} {
    _data = _mr->allocate(_size, _stream);
    assert(cudaSuccess == cudaMemcpyAsync(_data, other._data, _size,
                                          cudaMemcpyDefault, _stream));
  }

  /**---------------------------------------------------------------------------*
   * @brief Move constructor
   *
   * @param other
   *---------------------------------------------------------------------------**/
  device_buffer(device_buffer&& other)
      : _data{other._data},
        _size{other._size},
        _stream{other._stream},
        _mr{other._mr} {
    other._data = nullptr;
    other._size = 0;
    other._stream = 0;
    other._mr = nullptr;
  }

  /**---------------------------------------------------------------------------*
   * @brief Copy assignment operator
   *
   * @param other
   * @return device_buffer&
   *---------------------------------------------------------------------------**/
  device_buffer& operator=(device_buffer const& other) {
    if (&other != this) {
      _mr->deallocate(_data, _size, _stream);
      _size = other._size;
      _mr = other._mr;
      _stream = other._stream;
      _data = _mr->allocate(_size, _stream);
      assert(cudaSuccess == cudaMemcpyAsync(_data, other._data, _size,
                                            cudaMemcpyDefault, _stream));
    }
    return *this;
  }

  /**---------------------------------------------------------------------------*
   * @brief Move assignment operator
   *
   * @param other
   * @return device_buffer&
   *---------------------------------------------------------------------------**/
  device_buffer& operator=(device_buffer&& other) {
    if (&other != this) {
      _mr->deallocate(_data, _size, _stream);
      _data = other._data;
      _size = other._size;
      _stream = other._stream;
      _mr = other._mr;

      other._data = nullptr;
      other._size = 0;
      other._stream = 0;
      other._mr = nullptr;
    }
    return *this;
  }

  ~device_buffer() {
    _mr->deallocate(_data, _size, _stream);
    _data = nullptr;
    _size = 0;
    _stream = 0;
    _mr = nullptr;
  }

  /**---------------------------------------------------------------------------*
   * @brief Returns raw pointer to underlying device memory allocation
   *---------------------------------------------------------------------------**/
  void const* data() const { return _data; }

  /**---------------------------------------------------------------------------*
   * @brief Returns raw pointer to underlying device memory allocation
   *---------------------------------------------------------------------------**/
  void* data() { return _data; }

  /**---------------------------------------------------------------------------*
   * @brief Returns size in bytes of device memory allocation
   *---------------------------------------------------------------------------**/
  std::size_t size() const { return _size; }

  /**---------------------------------------------------------------------------*
   * @brief Returns stream used for allocation/deallocation
   *---------------------------------------------------------------------------**/
  cudaStream_t stream() const { return _stream; }

  /**---------------------------------------------------------------------------*
   * @brief Returns pointer to the memory resource used to allocate and
   * deallocate the device memory
   *---------------------------------------------------------------------------**/
  mr::device_memory_resource const * memory_resource() const { return _mr; }

 private:
  void* _data{nullptr};     ///< Pointer to device memory allocation
  std::size_t _size{0};     ///< Size in bytes of device memory allocation
  cudaStream_t _stream{0};  ///< Stream which may be used for
                            ///< allocation/deallocation of device memory
  mr::device_memory_resource* _mr{
      mr::get_default_resource()};  ///< The memory resource used to
                                    ///< allocate/deallocate device memory
};
}  // namespace rmm