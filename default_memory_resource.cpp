#include "default_memory_resource.hpp"
#include "cuda_memory_resource.hpp"
#include "device_memory_resource.hpp"

#include <atomic>

namespace rmm {
namespace mr {
namespace {
inline device_memory_resource* default_resource() {
  static cuda_memory_resource resource{};
  return &resource;
}

inline std::atomic<device_memory_resource*>& get_default() {
  static std::atomic<device_memory_resource*> res{default_resource()};
  return res;
}
}  // namespace

device_memory_resource* get_default_resource() { return get_default().load(); }

device_memory_resource* set_global_resource(
    device_memory_resource* new_resource) {
  return get_default().exchange(new_resource);
}
}  // namespace mr
}  // namespace rmm

namespace detail {}