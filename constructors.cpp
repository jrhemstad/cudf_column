/**---------------------------------------------------------------------------*
 * @brief
 * Assume that `DType` is the same as `gdf_dtype`, and that `device_buffer` is
 * effectively a `rmm::device_vector`.
 * Assume `bitmask` is a class that wraps a `device_buffer` of bitmask elements.
 *
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
 * @param[in] type The element type
 * @param[in] size The number of elements in the column
 * @param[in] allocate_bitmask Optionally allocate an appropriate sized
 * bitmask
 *---------------------------------------------------------------------------**/
column(DType type, int size, bool allocate_bitmask = false);

/**---------------------------------------------------------------------------*
 * @brief Construct a new column from a type, and a device_buffer for
 * data that will be *deep* copied.
 *
 * @param[in] dtype The element type
 * @param[in] size The number of elements in the column
 * @param[in] data_buffer device_buffer whose data will be *deep* copied
 *---------------------------------------------------------------------------**/
column(DType dtype, int size, device_buffer data);

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
 * @param[in] size The number of elements in the column
 * @param data device_buffer whose data will be moved from into this column
 * @param mask bitmask whose data will be moved into this column
 *---------------------------------------------------------------------------**/
column(DType dtype, int size, device_buffer&& data, bitmask&& mask);

/**---------------------------------------------------------------------------*
 * @brief Construct a new column from a type, size, and deep copied device
 * buffer for data, and moved bitmask.
 *
 * @param dtype The element type
 * @param size The number of elements 
 * @param data device_buffer whose data will be *deep* copied
 * @param mask bitmask whose data will be moved into this column
 *---------------------------------------------------------------------------**/
column(DType dtype, int size, device_buffer data, bitmask&& mask);

/**---------------------------------------------------------------------------*
 * @brief Construct a new column from a type, size, and moved device
 * buffer for data, and deep copied bitmask.
 *
 * @param dtype The element type
 * @param size The number of elements 
 * @param data device_buffer whose data will be moved into this column
 * @param mask bitmask whose data will be deep copied into this column
 *---------------------------------------------------------------------------**/
column(DType dtype, int size, device_buffer&& data, bitmask mask);

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
