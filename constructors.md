# Constructors

Assume that `DType` is the same as `gdf_dtype`, and that `device_buffer` is effectively a `rmm::device_vector`.

Note: This design document purposefully ignores the issue of `allocators` and controlling how the device memory for the column is allocated. This will be added as the allocator design in fleshed out. For the timebeing, assume that an allocator handle can be passed into each of the constructors that controls how the device memory for the data and bitmask is allocated.

```
/**---------------------------------------------------------------------------*
 * @brief Construct a new column from a size, type, and option to
 * allocate bitmask.
 * 
 * Both the data and bitmask are unintialized.
 *
 * @param[in] size The number of elements in the column
 * @param[in] type The element type
 * @param[in] allocate_bitmask Optionally allocate an appropriate sized
 * bitmask
 *---------------------------------------------------------------------------**/
column(int size, DType type, bool allocate_bitmask = false);

/**---------------------------------------------------------------------------*
 * @brief Construct a new column from a type, and device_buffers for
 * data and bitmask that will be *deep* copied.
 *
 * @param[in] dtype The element type
 * @param[in] data_buffer device_buffer whose data will be *deep* copied
 * @param[in] mask_buffer Optional device_buffer whose data will be *deep*
 *copied
 *---------------------------------------------------------------------------**/
column(DType dtype, device_buffer data,
       device_buffer mask_buffer = device_buffer{});

/**---------------------------------------------------------------------------*
 * @brief Construct a new column from a type, and device_buffers for data and
 * bitmask that will be *shallow* copied.
 *
 * This constructor uses move semantics to take ownership of the device_buffer's
 * device memory. The `device_buffer` passed into this constructor will not
 * longer be valid to use. Furthermore, it will result in undefined behavior if
 * the device_buffer`s associated memory is modified or freed after invoking
 * this constructor.
 *
 * @param dtype The element type
 * @param data device_buffer whose data will be moved from into this column
 * @param mask Optional device_buffer whose data will be moved from into this
 * column. If no device_buffer is passed in, it is assumed all elements are
 * valid and no bitmask is allocated.
 *---------------------------------------------------------------------------**/
column(DType dtype, device_buffer&& data,
       device_buffer&& mask_buffer = device_buffer{});

/**---------------------------------------------------------------------------*
 * @brief Construct a new column from a type, and a device_buffer for data.
 *
 * @note If `dtype` is a floating point type, an associated bitmask will be
 * created where every NaN value is set to NULL. Otherwise, the bitmask will not
 * be allocated.
 *
 * @param dtype The element type
 * @param data device_buffer whose data will be deep copied into this column
 * @param nan_as_null If `dtype` is a floating point type, optionally allocate a
 * bitmask where every NaN value is set to NULL.
 *---------------------------------------------------------------------------**/
column(DType dtype, device_buffer data, bool nan_as_null = true);

/**---------------------------------------------------------------------------*
 * @brief Construct a new column from a type, and a device_buffer whose device
 * memory will be moved into this column.
 *
 * @note If `dtype` is a floating point type, an associated bitmask will be
 * created where every NaN value is set to NULL. Otherwise, the bitmask will not
 * be allocated.
 *
 * @param dtype The element type
 * @param data device_buffer whose data will be moved into this column
 * @param nan_as_null If `dtype` is a floating point type, optionally allocate a
 * bitmask where every NaN value is set to NULL.
 *---------------------------------------------------------------------------**/
column(DType dtype, device_buffer&& data, bool nan_as_null = true);

/**---------------------------------------------------------------------------*
 * @brief Construct a new column from a type, and host_buffers for
 * data and bitmask.
 *
 * New device memory allocations will be made for the data and bitmask and the
 * host_buffer's contents will be copied to device.
 *
 * @param[in] dtype The element type
 * @param[in] data_buffer device_buffer whose data will be *deep* copied
 * @param[in] mask_buffer Optional device_buffer whose data will be *deep*
 *copied
 *---------------------------------------------------------------------------**/
column(DType dtype, host_buffer const& data,
       host_buffer const& mask_buffer = host_buffer{});

/**---------------------------------------------------------------------------*
 * @brief Construct a new column from a type, and a host_buffer for data.
 *
 * New device memory allocations will be made for the data and the
 * host_buffer's contents will be copied to device.
 *
 * @note If `dtype` is a floating point type, an associated bitmask will be
 * created where every NaN value is set to NULL. Otherwise, the bitmask will not
 * be allocated.
 *
 * @param dtype The element type
 * @param data device_buffer whose data will be deep copied into this column
 * @param nan_as_null If `dtype` is a floating point type, optionally allocate a
 * bitmask where every NaN value is set to NULL.
 *---------------------------------------------------------------------------**/
column(DType dtype, host_buffer const& data, bool nan_as_null = true);
```
