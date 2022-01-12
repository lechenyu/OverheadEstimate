
// Copyright (c) 2015-16 Tom Deakin, Simon McIntosh-Smith,
// University of Bristol HPC
//
// For full license terms please see the LICENSE file distributed with this
// source code


#include "CUDAStream.h"

void check_error(void)
{
  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess)
  {
    std::cerr << "Error: " << cudaGetErrorString(err) << std::endl;
    exit(err);
  }
}

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

#ifdef ESTIMATE
  cudaMalloc(&sa, ARRAY_SIZE*sizeof(uint32_t));
  check_error();
  cudaMalloc(&sb, ARRAY_SIZE*sizeof(uint32_t));
  check_error();
  cudaMalloc(&sc, ARRAY_SIZE*sizeof(uint32_t));
  check_error();
  cudaMalloc(&ssum, DOT_NUM_BLOCKS*sizeof(uint32_t));
  check_error();
#endif

#endif


#ifdef ESTIMATE
  std::cout << "Estimate Version\n";
#else
  std::cout << "Original Version\n";
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

#ifdef ESTIMATE
  cudaFree(sa);
  check_error();
  cudaFree(sb);
  check_error();
  cudaFree(sc);
  check_error();
  cudaFree(ssum);
  check_error();
#endif

#endif
}


template <typename T>
#ifdef ESTIMATE
__global__ void init_kernel(T * a, T * b, T * c, T initA, T initB, T initC, uint32_t *sa, uint32_t *sb, uint32_t *sc)
#else
__global__ void init_kernel(T * a, T * b, T * c, T initA, T initB, T initC)
#endif
{
  const int i = blockDim.x * blockIdx.x + threadIdx.x;
  a[i] = initA;
  b[i] = initB;
  c[i] = initC;

#ifdef ESTIMATE   
  record_w(sa, i)
  record_w(sb, i)
  record_w(sc, i)
#endif
}

template <class T>
void CUDAStream<T>::init_arrays(T initA, T initB, T initC)
{
#ifdef ESTIMATE
  init_kernel<<<array_size/TBSIZE, TBSIZE>>>(d_a, d_b, d_c, initA, initB, initC, sa, sb, sc);
#else
  init_kernel<<<array_size/TBSIZE, TBSIZE>>>(d_a, d_b, d_c, initA, initB, initC);
#endif
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
__global__ void copy_kernel(const T * a, T * c, uint32_t *sa, uint32_t *sc)
#else
__global__ void copy_kernel(const T * a, T * c)
#endif
{
  const int i = blockDim.x * blockIdx.x + threadIdx.x;
  c[i] = a[i];

#ifdef ESTIMATE
    record_r(sa, i)
    record_w(sc, i)
#endif  
}

template <class T>
void CUDAStream<T>::copy()
{
#ifdef ESTIMATE
  copy_kernel<<<array_size/TBSIZE, TBSIZE>>>(d_a, d_c, sa, sc);
#else
  copy_kernel<<<array_size/TBSIZE, TBSIZE>>>(d_a, d_c);
#endif
  check_error();
  cudaDeviceSynchronize();
  check_error();
}

template <typename T>
#ifdef ESTIMATE
__global__ void mul_kernel(T * b, const T * c, uint32_t *sb, uint32_t *sc)
#else
__global__ void mul_kernel(T * b, const T * c)
#endif
{
  const T scalar = startScalar;
  const int i = blockDim.x * blockIdx.x + threadIdx.x;
  b[i] = scalar * c[i];

#ifdef ESTIMATE
    record_r(sc, i)
    record_w(sb, i)
#endif 
}

template <class T>
void CUDAStream<T>::mul()
{
#ifdef ESTIMATE
  mul_kernel<<<array_size/TBSIZE, TBSIZE>>>(d_b, d_c, sb, sc);
#else
  mul_kernel<<<array_size/TBSIZE, TBSIZE>>>(d_b, d_c);
#endif
  check_error();
  cudaDeviceSynchronize();
  check_error();
}

template <typename T>
#ifdef ESTIMATE
__global__ void add_kernel(const T * a, const T * b, T * c, uint32_t *sa, uint32_t *sb, uint32_t *sc)
#else
__global__ void add_kernel(const T * a, const T * b, T * c)
#endif
{
  const int i = blockDim.x * blockIdx.x + threadIdx.x;
  c[i] = a[i] + b[i];

#ifdef ESTIMATE
    record_r(sa, i)
    record_r(sb, i)
    record_w(sc, i)
#endif
}

template <class T>
void CUDAStream<T>::add()
{
#ifdef ESTIMATE
  add_kernel<<<array_size/TBSIZE, TBSIZE>>>(d_a, d_b, d_c, sa, sb, sc);
#else  
  add_kernel<<<array_size/TBSIZE, TBSIZE>>>(d_a, d_b, d_c);
#endif
  check_error();
  cudaDeviceSynchronize();
  check_error();
}

template <typename T>
#ifdef ESTIMATE
__global__ void triad_kernel(T * a, const T * b, const T * c, uint32_t *sa, uint32_t *sb, uint32_t *sc)
#else
__global__ void triad_kernel(T * a, const T * b, const T * c)
#endif
{
  const T scalar = startScalar;
  const int i = blockDim.x * blockIdx.x + threadIdx.x;
  a[i] = b[i] + scalar * c[i];

#ifdef ESTIMATE
    record_r(sb, i)
    record_r(sc, i)
    record_w(sa, i)
#endif
}

template <class T>
void CUDAStream<T>::triad()
{
#ifdef ESTIMATE
  triad_kernel<<<array_size/TBSIZE, TBSIZE>>>(d_a, d_b, d_c, sa, sb, sc);
#else
  triad_kernel<<<array_size/TBSIZE, TBSIZE>>>(d_a, d_b, d_c);
#endif
  check_error();
  cudaDeviceSynchronize();
  check_error();
}

template <typename T>
#ifdef ESTIMATE
__global__ void nstream_kernel(T * a, const T * b, const T * c, uint32_t *sa, uint32_t *sb, uint32_t *sc)
#else
__global__ void nstream_kernel(T * a, const T * b, const T * c)
#endif
{
  const T scalar = startScalar;
  const int i = blockDim.x * blockIdx.x + threadIdx.x;
  a[i] += b[i] + scalar * c[i];

#ifdef ESTIMATE
    record_r(sb, i)
    record_r(sc, i)
    record_w(sa, i)
#endif
}

template <class T>
void CUDAStream<T>::nstream()
{
#ifdef ESTIMATE
  nstream_kernel<<<array_size/TBSIZE, TBSIZE>>>(d_a, d_b, d_c, sa, sb, sc);
#else
  nstream_kernel<<<array_size/TBSIZE, TBSIZE>>>(d_a, d_b, d_c);
#endif
  check_error();
  cudaDeviceSynchronize();
  check_error();
}

template <class T>
#ifdef ESTIMATE
__global__ void dot_kernel(const T * a, const T * b, T * sum, int array_size, uint32_t *sa, uint32_t *sb, uint32_t *ssum)
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
    record_r(sa, i)
    record_r(sb, i)
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
    record_w(ssum, blockIdx.x)
#endif
  }
}

template <class T>
T CUDAStream<T>::dot()
{
#ifdef ESTIMATE
  dot_kernel<<<DOT_NUM_BLOCKS, TBSIZE>>>(d_a, d_b, d_sum, array_size, sa, sb, ssum);
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
