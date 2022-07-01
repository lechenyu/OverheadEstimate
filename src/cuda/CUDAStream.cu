
// Copyright (c) 2015-16 Tom Deakin, Simon McIntosh-Smith,
// University of Bristol HPC
//
// For full license terms please see the LICENSE file distributed with this
// source code


#include "CUDAStream.h"
#ifdef ESTIMATE
#include "rle.h"
#endif

#define check_error() { check_error_impl(__FILE__, __LINE__); }
void check_error_impl(const char *file, int line)
{
  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess)
  {
    std::cerr << "Error at " << file << " : line " << line << " >>> " << cudaGetErrorString(err) << std::endl;
    exit(err);
  }
}

#ifdef ESTIMATE
__global__ void extract_three_bits(uint8_t *dest, shadow_t *src, size_t length) {
  const int i = blockDim.x * blockIdx.x + threadIdx.x;
  if (i < length / 2) {
    const int left = 2 * i;
    const int right = 2 * i + 1;
    dest[i] = ((src[left] & 0x00000007) << 4) | (src[right] & 0x00000007);
  }
}

void compress_and_memcpy(shadow_t *host, shadow_t *device, size_t length) {
  uint8_t *compressed;
  int block_num = (length + 2*TBSIZE - 1)/(2*TBSIZE);
  cudaMalloc(&compressed, length / 2);
  check_error();
  extract_three_bits<<<block_num, TBSIZE>>>(compressed, device, length);
  check_error();
  cudaMemcpy(host, compressed, length / 2, cudaMemcpyDeviceToHost);
  check_error();
  cudaFree(compressed);
  check_error();
}

size_t rle_and_memcpy(uint8_t *host_value, int *host_index, shadow_t *device, size_t length) {
  uint8_t *compressed;
  int block_num = (length + 2*TBSIZE - 1)/(2*TBSIZE);
  cudaMalloc(&compressed, length / 2);
  check_error();
  extract_three_bits<<<block_num, TBSIZE>>>(compressed, device, length);
  check_error();
  RleResult<uint8_t> &&d_result = parallel_rle_impl(compressed, length / 2);
  cudaMemcpy(host_value, d_result.value, d_result.len * sizeof(uint8_t), cudaMemcpyDeviceToHost);
  check_error();
  cudaMemcpy(host_index, d_result.index, d_result.len * sizeof(int), cudaMemcpyDeviceToHost);
  check_error();
  cudaFree(compressed);
  check_error();
  return d_result.len;
}

template<typename T>
void verify_shadow(std::string name, T *s, size_t length, T val) {
  int i;
  for (i = 0; i < length && s[i] == val; i++) {}
  if (i == length) {
    // std::cout << name << " passes verification\n";
  } else {
    std::cout << name << "[" << i << "] = " << s[i] << ", expect " << val << "\n";
    exit(1);
  }
}
#endif

template <class T>
CUDAStream<T>::CUDAStream(const int ARRAY_SIZE, const int device_index)
{

  // The array size must be divisible by TBSIZE for kernel launches
  if (ARRAY_SIZE % TBSIZE != 0)
  {
    std::stringstream ss;
    ss << "Array size must be a multiple of " << TBSIZE;
    throw std::runtime_error(ss.str());
  }

  // Set device
  int count;
  cudaGetDeviceCount(&count);
  check_error();
  if (device_index >= count)
    throw std::runtime_error("Invalid device index");
  cudaSetDevice(device_index);
  check_error();

  // Print out device information
  std::cout << "Using CUDA device " << getDeviceName(device_index) << std::endl;
  std::cout << "Driver: " << getDeviceDriver(device_index) << std::endl;

  array_size = ARRAY_SIZE;

  // Allocate the host array for partial sums for dot kernels
  sums = (T*)malloc(sizeof(T) * DOT_NUM_BLOCKS);

  // Check buffers fit on the device
  cudaDeviceProp props;
  cudaGetDeviceProperties(&props, 0);
  if (props.totalGlobalMem < 3*ARRAY_SIZE*sizeof(T))
    throw std::runtime_error("Device does not have enough memory for all 3 buffers");

  // Create device buffers
#if defined(MANAGED)
  cudaMallocManaged(&d_a, ARRAY_SIZE*sizeof(T));
  check_error();
  cudaMallocManaged(&d_b, ARRAY_SIZE*sizeof(T));
  check_error();
  cudaMallocManaged(&d_c, ARRAY_SIZE*sizeof(T));
  check_error();
  cudaMallocManaged(&d_sum, DOT_NUM_BLOCKS*sizeof(T));
  check_error();
#elif defined(PAGEFAULT)
  d_a = (T*)malloc(sizeof(T)*ARRAY_SIZE);
  d_b = (T*)malloc(sizeof(T)*ARRAY_SIZE);
  d_c = (T*)malloc(sizeof(T)*ARRAY_SIZE);
  d_sum = (T*)malloc(sizeof(T)*DOT_NUM_BLOCKS);
#else
  cudaMalloc(&d_a, ARRAY_SIZE*sizeof(T));
  check_error();
  cudaMalloc(&d_b, ARRAY_SIZE*sizeof(T));
  check_error();
  cudaMalloc(&d_c, ARRAY_SIZE*sizeof(T));
  check_error();
  cudaMalloc(&d_sum, DOT_NUM_BLOCKS*sizeof(T));
  check_error();
#endif

#if ESTIMATE == EXPLICIT
  cudaMalloc(&d_sa, ARRAY_SIZE * sizeof(shadow_t));
  check_error();
  cudaMalloc(&d_sb, ARRAY_SIZE * sizeof(shadow_t));
  check_error();
  cudaMalloc(&d_sc, ARRAY_SIZE * sizeof(shadow_t));
  check_error();
  cudaMalloc(&d_ssum, DOT_NUM_BLOCKS * sizeof(shadow_t));
  check_error();
  sa = (shadow_t *)malloc(ARRAY_SIZE * sizeof(shadow_t));
  sb = (shadow_t *)malloc(ARRAY_SIZE * sizeof(shadow_t));
  sc = (shadow_t *)malloc(ARRAY_SIZE * sizeof(shadow_t));
  ssum = (shadow_t *)malloc(DOT_NUM_BLOCKS * sizeof(shadow_t));
#if OPTIMIZE == RLE
  value_sa = (uint8_t *)malloc(ARRAY_SIZE * sizeof(uint8_t));
  value_sb = (uint8_t *)malloc(ARRAY_SIZE * sizeof(uint8_t));
  value_sc = (uint8_t *)malloc(ARRAY_SIZE * sizeof(uint8_t));
  value_ssum = (uint8_t *)malloc(DOT_NUM_BLOCKS * sizeof(uint8_t));
  index_sa = (int *)malloc(ARRAY_SIZE * sizeof(int));
  index_sb = (int *)malloc(ARRAY_SIZE * sizeof(int));
  index_sc = (int *)malloc(ARRAY_SIZE * sizeof(int));
  index_ssum = (int *)malloc(DOT_NUM_BLOCKS * sizeof(int));
#endif

#elif ESTIMATE == MANAGED
  cudaMallocManaged(&d_sa, ARRAY_SIZE * sizeof(shadow_t));
  check_error();
  cudaMallocManaged(&d_sb, ARRAY_SIZE * sizeof(shadow_t));
  check_error();
  cudaMallocManaged(&d_sc, ARRAY_SIZE * sizeof(shadow_t));
  check_error();
  cudaMallocManaged(&d_ssum, DOT_NUM_BLOCKS * sizeof(shadow_t));
  check_error();
  sa = d_sa;
  sb = d_sb;
  sc = d_sc;
  ssum = d_ssum;
#elif ESTIMATE == PINNED
  unsigned int flags = cudaHostAllocDefault;
  cudaHostAlloc(&sa, ARRAY_SIZE * sizeof(shadow_t), flags);
  check_error();
  cudaHostAlloc(&sb, ARRAY_SIZE * sizeof(shadow_t), flags);
  check_error();
  cudaHostAlloc(&sc, ARRAY_SIZE * sizeof(shadow_t), flags);
  check_error();
  cudaHostAlloc(&ssum, DOT_NUM_BLOCKS * sizeof(shadow_t), flags);
  check_error();
  d_sa = sa;
  d_sb = sb;
  d_sc = sc;
  d_ssum = ssum;
#endif


#ifdef ESTIMATE
#if ESTIMATE == EXPLICIT
  std::cout << "Explicit device memory version\n";
#elif ESTIMATE == MANAGED
  std::cout << "Unified memory version\n";
#elif ESTIMATE == PINNED
  std::cout << "Page-locked host memory version\n";
#else
  std::cout << "Unknown ESTIMATE type\n";
  exit(1);
#endif // end of #if ESTIMATE
#else
  std::cout << "Original version\n";
#endif
}


template <class T>
CUDAStream<T>::~CUDAStream()
{
  free(sums);

#if defined(PAGEFAULT)
  free(d_a);
  free(d_b);
  free(d_c);
  free(d_sum);
#else
  cudaFree(d_a);
  check_error();
  cudaFree(d_b);
  check_error();
  cudaFree(d_c);
  check_error();
  cudaFree(d_sum);
  check_error();
#endif

#if ESTIMATE == EXPLICIT || ESTIMATE == MANAGED
  cudaFree(d_sa);
  check_error();
  cudaFree(d_sb);
  check_error();
  cudaFree(d_sc);
  check_error();
  cudaFree(d_ssum);
  check_error();
#elif ESTIMATE == PINNED
  cudaFreeHost(sa);
  check_error();
  cudaFreeHost(sb);
  check_error();
  cudaFreeHost(sc);
  check_error();
  cudaFreeHost(ssum);
  check_error();
#endif
}


template <typename T>
__global__ void init_kernel(T * a, T * b, T * c, T initA, T initB, T initC)
{
  const int i = blockDim.x * blockIdx.x + threadIdx.x;
  a[i] = initA;
  b[i] = initB;
  c[i] = initC;
}

template <class T>
void CUDAStream<T>::init_arrays(T initA, T initB, T initC)
{
  init_kernel<<<array_size/TBSIZE, TBSIZE>>>(d_a, d_b, d_c, initA, initB, initC);
  check_error();
  cudaDeviceSynchronize();
  check_error();
}

template <class T>
void CUDAStream<T>::read_arrays(std::vector<T>& a, std::vector<T>& b, std::vector<T>& c)
{
  // Copy device memory to host
#if defined(PAGEFAULT) || defined(MANAGED)
  cudaDeviceSynchronize();
  for (int i = 0; i < array_size; i++)
  {
    a[i] = d_a[i];
    b[i] = d_b[i];
    c[i] = d_c[i];
  }
#else
  cudaMemcpy(a.data(), d_a, a.size()*sizeof(T), cudaMemcpyDeviceToHost);
  check_error();
  cudaMemcpy(b.data(), d_b, b.size()*sizeof(T), cudaMemcpyDeviceToHost);
  check_error();
  cudaMemcpy(c.data(), d_c, c.size()*sizeof(T), cudaMemcpyDeviceToHost);
  check_error();
#endif
}


template <typename T>
#ifdef ESTIMATE
__global__ void copy_kernel(const T * a, T * c, uint32_t *d_sa, uint32_t *d_sc)
#else
__global__ void copy_kernel(const T * a, T * c)
#endif
{
  const int i = blockDim.x * blockIdx.x + threadIdx.x;
  c[i] = a[i];

#ifdef ESTIMATE
  record_r(d_sa, i)
  record_w(d_sc, i)
#endif  
}

template <class T>
void CUDAStream<T>::copy()
{
#ifdef ESTIMATE
#if ESTIMATE == EXPLICIT || ESTIMATE == MANAGED
  cudaMemset(d_sa, 0, array_size * sizeof(shadow_t));
  cudaMemset(d_sc, 0, array_size * sizeof(shadow_t));
#elif ESTIMATE == PINNED
  memset(sa, 0, array_size * sizeof(shadow_t));
  memset(sc, 0, array_size * sizeof(shadow_t));
#endif
  copy_kernel<<<array_size/TBSIZE, TBSIZE>>>(d_a, d_c, d_sa, d_sc);
#else
  copy_kernel<<<array_size/TBSIZE, TBSIZE>>>(d_a, d_c);
#endif
  check_error();  
  cudaDeviceSynchronize();
  check_error();

#if ESTIMATE == EXPLICIT
#if OPTIMIZE == NONE
  cudaMemcpy(sa, d_sa, array_size * sizeof(shadow_t), cudaMemcpyDeviceToHost);
  check_error();
  cudaMemcpy(sc, d_sc, array_size * sizeof(shadow_t), cudaMemcpyDeviceToHost);
  check_error();
#elif OPTIMIZE == NOPC
  compress_and_memcpy(sa, d_sa, array_size);
  compress_and_memcpy(sc, d_sc, array_size);
#elif OPTIMIZE == RLE
  size_t len_sa = rle_and_memcpy(value_sa, index_sa, d_sa, array_size);
  size_t len_sc = rle_and_memcpy(value_sc, index_sc, d_sc, array_size);
  // for (int i = 0; i < len_sa; i++) {
  //   std::cout << index_sa[i] << ":" << (unsigned)value_sa[i] << " ";
  // }
  // std::cout << std::endl;
  // for (int i = 0; i < len_sc; i++) {
  //   std::cout << index_sc[i] << ":" << (unsigned)value_sc[i] << " ";
  // }
  // std::cout << std::endl;
#elif OPTIMIZE == NOMEMCPY

#endif // end of #if OPTIMIZE
#endif

// #ifdef ESTIMATE
// #if OPTIMIZE == NONE
//   verify_shadow<shadow_t>("sa", sa, array_size, 0x00000005);
//   verify_shadow<shadow_t>("sc", sc, array_size, 0x00000002);
// #elif OPTIMIZE == NOPC
//   verify_shadow<uint8_t>("sa", (uint8_t *)sa, array_size / 2, 0x55);
//   verify_shadow<uint8_t>("sc", (uint8_t *)sc, array_size / 2, 0x22);
// #endif // end of #if OPTIMIZE
// #endif
}

template <typename T>
#ifdef ESTIMATE
__global__ void mul_kernel(T * b, const T * c, uint32_t *d_sb, uint32_t *d_sc)
#else
__global__ void mul_kernel(T * b, const T * c)
#endif
{
  const T scalar = startScalar;
  const int i = blockDim.x * blockIdx.x + threadIdx.x;
  b[i] = scalar * c[i];

#ifdef ESTIMATE
    record_r(d_sc, i)
    record_w(d_sb, i)
#endif 
}

template <class T>
void CUDAStream<T>::mul()
{
#ifdef ESTIMATE
#if ESTIMATE == EXPLICIT || ESTIMATE == MANAGED
  cudaMemset(d_sb, 0, array_size * sizeof(shadow_t));
  cudaMemset(d_sc, 0, array_size * sizeof(shadow_t));
#elif ESTIMATE == PINNED
  memset(sb, 0, array_size * sizeof(shadow_t));
  memset(sc, 0, array_size * sizeof(shadow_t));
#endif
  mul_kernel<<<array_size/TBSIZE, TBSIZE>>>(d_b, d_c, d_sb, d_sc);
#else
  mul_kernel<<<array_size/TBSIZE, TBSIZE>>>(d_b, d_c);
#endif
  check_error();
  cudaDeviceSynchronize();
  check_error();

#if ESTIMATE == EXPLICIT
#if OPTIMIZE == NONE
  cudaMemcpy(sb, d_sb, array_size * sizeof(shadow_t), cudaMemcpyDeviceToHost);
  check_error();
  cudaMemcpy(sc, d_sc, array_size * sizeof(shadow_t), cudaMemcpyDeviceToHost);
  check_error();
#elif OPTIMIZE == NOPC
  compress_and_memcpy(sb, d_sb, array_size);
  compress_and_memcpy(sc, d_sc, array_size);
#elif OPTIMIZE == RLE
  size_t len_sb = rle_and_memcpy(value_sb, index_sb, d_sb, array_size);
  size_t len_sc = rle_and_memcpy(value_sc, index_sc, d_sc, array_size);
#elif OPTIMIZE == NOMEMCPY

#endif // end of #if OPTIMIZE
#endif
}

template <typename T>
#ifdef ESTIMATE
__global__ void add_kernel(const T * a, const T * b, T * c, uint32_t *d_sa, uint32_t *d_sb, uint32_t *d_sc)
#else
__global__ void add_kernel(const T * a, const T * b, T * c)
#endif
{
  const int i = blockDim.x * blockIdx.x + threadIdx.x;
  c[i] = a[i] + b[i];

#ifdef ESTIMATE
    record_r(d_sa, i)
    record_r(d_sb, i)
    record_w(d_sc, i)
#endif
}

template <class T>
void CUDAStream<T>::add()
{
#ifdef ESTIMATE
#if ESTIMATE == EXPLICIT || ESTIMATE == MANAGED
  cudaMemset(d_sa, 0, array_size * sizeof(shadow_t));
  cudaMemset(d_sb, 0, array_size * sizeof(shadow_t));
  cudaMemset(d_sc, 0, array_size * sizeof(shadow_t));
#elif ESTIMATE == PINNED
  memset(sa, 0, array_size * sizeof(shadow_t));
  memset(sb, 0, array_size * sizeof(shadow_t));
  memset(sc, 0, array_size * sizeof(shadow_t));
#endif
  add_kernel<<<array_size/TBSIZE, TBSIZE>>>(d_a, d_b, d_c, d_sa, d_sb, d_sc);
#else  
  add_kernel<<<array_size/TBSIZE, TBSIZE>>>(d_a, d_b, d_c);
#endif
  check_error();
  cudaDeviceSynchronize();
  check_error();

#if ESTIMATE == EXPLICIT
#if OPTIMIZE == NONE
  cudaMemcpy(sa, d_sa, array_size * sizeof(shadow_t), cudaMemcpyDeviceToHost);
  check_error();
  cudaMemcpy(sb, d_sb, array_size * sizeof(shadow_t), cudaMemcpyDeviceToHost);
  check_error();
  cudaMemcpy(sc, d_sc, array_size * sizeof(shadow_t), cudaMemcpyDeviceToHost);
  check_error();
#elif OPTIMIZE == NOPC
  compress_and_memcpy(sa, d_sa, array_size);
  compress_and_memcpy(sb, d_sb, array_size);
  compress_and_memcpy(sc, d_sc, array_size);
#elif OPTIMIZE == RLE
  size_t len_sa = rle_and_memcpy(value_sa, index_sa, d_sa, array_size);
  size_t len_sb = rle_and_memcpy(value_sb, index_sb, d_sb, array_size);
  size_t len_sc = rle_and_memcpy(value_sc, index_sc, d_sc, array_size);
#elif OPTIMIZE == NOMEMCPY

#endif // end of #if OPTIMIZE
#endif
}

template <typename T>
#ifdef ESTIMATE
__global__ void triad_kernel(T * a, const T * b, const T * c, uint32_t *d_sa, uint32_t *d_sb, uint32_t *d_sc)
#else
__global__ void triad_kernel(T * a, const T * b, const T * c)
#endif
{
  const T scalar = startScalar;
  const int i = blockDim.x * blockIdx.x + threadIdx.x;
  a[i] = b[i] + scalar * c[i];

#ifdef ESTIMATE
    record_r(d_sb, i)
    record_r(d_sc, i)
    record_w(d_sa, i)
#endif
}

template <class T>
void CUDAStream<T>::triad()
{
#ifdef ESTIMATE
#if ESTIMATE == EXPLICIT || ESTIMATE == MANAGED
  cudaMemset(d_sa, 0, array_size * sizeof(shadow_t));
  cudaMemset(d_sb, 0, array_size * sizeof(shadow_t));
  cudaMemset(d_sc, 0, array_size * sizeof(shadow_t));
#elif ESTIMATE == PINNED
  memset(sa, 0, array_size * sizeof(shadow_t));
  memset(sb, 0, array_size * sizeof(shadow_t));
  memset(sc, 0, array_size * sizeof(shadow_t));
#endif
  triad_kernel<<<array_size/TBSIZE, TBSIZE>>>(d_a, d_b, d_c, d_sa, d_sb, d_sc);
#else
  triad_kernel<<<array_size/TBSIZE, TBSIZE>>>(d_a, d_b, d_c);
#endif
  check_error();
  cudaDeviceSynchronize();
  check_error();

#if ESTIMATE == EXPLICIT
#if OPTIMIZE == NONE
  cudaMemcpy(sa, d_sa, array_size * sizeof(shadow_t), cudaMemcpyDeviceToHost);
  check_error();
  cudaMemcpy(sb, d_sb, array_size * sizeof(shadow_t), cudaMemcpyDeviceToHost);
  check_error();
  cudaMemcpy(sc, d_sc, array_size * sizeof(shadow_t), cudaMemcpyDeviceToHost);
  check_error();
#elif OPTIMIZE == NOPC
  compress_and_memcpy(sa, d_sa, array_size);
  compress_and_memcpy(sb, d_sb, array_size);
  compress_and_memcpy(sc, d_sc, array_size);
#elif OPTIMIZE == RLE
  size_t len_sa = rle_and_memcpy(value_sa, index_sa, d_sa, array_size);
  size_t len_sb = rle_and_memcpy(value_sb, index_sb, d_sb, array_size);
  size_t len_sc = rle_and_memcpy(value_sc, index_sc, d_sc, array_size);
#elif OPTIMIZE == NOMEMCPY

#endif // end of #if OPTIMIZE
#endif
}

template <typename T>
#ifdef ESTIMATE
__global__ void nstream_kernel(T * a, const T * b, const T * c, uint32_t *d_sa, uint32_t *d_sb, uint32_t *d_sc)
#else
__global__ void nstream_kernel(T * a, const T * b, const T * c)
#endif
{
  const T scalar = startScalar;
  const int i = blockDim.x * blockIdx.x + threadIdx.x;
  a[i] += b[i] + scalar * c[i];

#ifdef ESTIMATE
    // record_r(d_sb, i)
    // record_r(d_sc, i)
    // record_w(d_sa, i)
#endif
}

template <class T>
void CUDAStream<T>::nstream()
{
#ifdef ESTIMATE
  nstream_kernel<<<array_size/TBSIZE, TBSIZE>>>(d_a, d_b, d_c, d_sa, d_sb, d_sc);
#else
  nstream_kernel<<<array_size/TBSIZE, TBSIZE>>>(d_a, d_b, d_c);
#endif
  check_error();
  cudaDeviceSynchronize();
  check_error();
}

template <class T>
#ifdef ESTIMATE
__global__ void dot_kernel(const T * a, const T * b, T * sum, int array_size, uint32_t *d_sa, uint32_t *d_sb, uint32_t *d_ssum)
#else
__global__ void dot_kernel(const T * a, const T * b, T * sum, int array_size)
#endif
{
  __shared__ T tb_sum[TBSIZE];

  int i = blockDim.x * blockIdx.x + threadIdx.x;
  const size_t local_i = threadIdx.x;

  tb_sum[local_i] = 0.0;
  for (; i < array_size; i += blockDim.x*gridDim.x) {
    tb_sum[local_i] += a[i] * b[i];
#ifdef ESTIMATE
    record_r(d_sa, i)
    record_r(d_sb, i)
#endif
  }

  for (int offset = blockDim.x / 2; offset > 0; offset /= 2)
  {
    __syncthreads();
    if (local_i < offset)
    {
      tb_sum[local_i] += tb_sum[local_i+offset];
    }
  }

  if (local_i == 0) {
    sum[blockIdx.x] = tb_sum[local_i];
#ifdef ESTIMATE
    record_w(d_ssum, blockIdx.x)
#endif
  }
}

template <class T>
T CUDAStream<T>::dot()
{
#ifdef ESTIMATE
#if ESTIMATE == EXPLICIT || ESTIMATE == MANAGED
  cudaMemset(d_sa, 0, array_size * sizeof(shadow_t));
  cudaMemset(d_sb, 0, array_size * sizeof(shadow_t));
  cudaMemset(d_ssum, 0, DOT_NUM_BLOCKS * sizeof(shadow_t));
#elif ESTIMATE == PINNED
  memset(sa, 0, array_size * sizeof(shadow_t));
  memset(sb, 0, array_size * sizeof(shadow_t));
  memset(ssum, 0, DOT_NUM_BLOCKS * sizeof(shadow_t));
#endif
  dot_kernel<<<DOT_NUM_BLOCKS, TBSIZE>>>(d_a, d_b, d_sum, array_size, d_sa, d_sb, d_ssum);
#else  
  dot_kernel<<<DOT_NUM_BLOCKS, TBSIZE>>>(d_a, d_b, d_sum, array_size);
#endif
  check_error();

#if defined(MANAGED) || defined(PAGEFAULT)
  cudaDeviceSynchronize();
  check_error();
#else
  cudaMemcpy(sums, d_sum, DOT_NUM_BLOCKS*sizeof(T), cudaMemcpyDeviceToHost);
  check_error();
#endif

  T sum = 0.0;
  for (int i = 0; i < DOT_NUM_BLOCKS; i++)
  {
#if defined(MANAGED) || defined(PAGEFAULT)
    sum += d_sum[i];
#else
    sum += sums[i];
#endif
  }

#if ESTIMATE == EXPLICIT
#if OPTIMIZE == NONE
  cudaMemcpy(sa, d_sa, array_size * sizeof(shadow_t), cudaMemcpyDeviceToHost);
  check_error();
  cudaMemcpy(sb, d_sb, array_size * sizeof(shadow_t), cudaMemcpyDeviceToHost);
  check_error();
  cudaMemcpy(ssum, d_ssum, DOT_NUM_BLOCKS * sizeof(shadow_t), cudaMemcpyDeviceToHost);
  check_error();
#elif OPTIMIZE == NOPC
  compress_and_memcpy(sa, d_sa, array_size);
  compress_and_memcpy(sb, d_sb, array_size);
  compress_and_memcpy(ssum, d_ssum, DOT_NUM_BLOCKS);
#elif OPTIMIZE == RLE
  size_t len_sa = rle_and_memcpy(value_sa, index_sa, d_sa, array_size);
  size_t len_sb = rle_and_memcpy(value_sb, index_sb, d_sb, array_size);
  size_t len_ssum = rle_and_memcpy(value_ssum, index_ssum, d_ssum, DOT_NUM_BLOCKS);
#elif OPTIMIZE == NOMEMCPY

#endif // end of #if OPTIMIZE
#endif
  return sum;
}

void listDevices(void)
{
  // Get number of devices
  int count;
  cudaGetDeviceCount(&count);
  check_error();

  // Print device names
  if (count == 0)
  {
    std::cerr << "No devices found." << std::endl;
  }
  else
  {
    std::cout << std::endl;
    std::cout << "Devices:" << std::endl;
    for (int i = 0; i < count; i++)
    {
      std::cout << i << ": " << getDeviceName(i) << std::endl;
    }
    std::cout << std::endl;
  }
}


std::string getDeviceName(const int device)
{
  cudaDeviceProp props;
  cudaGetDeviceProperties(&props, device);
  check_error();
  return std::string(props.name);
}


std::string getDeviceDriver(const int device)
{
  cudaSetDevice(device);
  check_error();
  int driver;
  cudaDriverGetVersion(&driver);
  check_error();
  return std::to_string(driver);
}

template class CUDAStream<float>;
template class CUDAStream<double>;
