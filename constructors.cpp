/**---------------------------------------------------------------------------*
 * @brief
 * Assume that `DType` is the same as `gdf_dtype`, and that `device_buffer` is
effectively a `rmm::device_vector`.
 *  Note: This design purposefully ignores the issue of `allocators` and
controlling how the device memory for the column is allocated. This will be
added as the allocator design in fleshed out. For the timebeing, assume that an
allocator handle can be passed into each of the constructors that controls how
the device memory for the data and bitmask is allocated.
 *
*---------------------------------------------------------------------------**/

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
 * @brief Construct a new column by deep copying the device memory of another
 * column.
 *
 * @param other The other column to copy
 *---------------------------------------------------------------------------**/
column(column const& other);

/**---------------------------------------------------------------------------*
 * @brief Construct a new column object by moving the device memory from another
 * column.
 *
 * @param other The other column whose device memory will be moved to the new
 * column
 *---------------------------------------------------------------------------**/
column(column&& other);
