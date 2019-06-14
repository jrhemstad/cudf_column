#include "device_memory_resource.hpp"
namespace rmm {
namespace mr {

device_memory_resource* get_default_resource();
device_memory_resource* set_global_resource(device_memory_resource* new_res);

}  // namespace mr
}  // namespace rmm