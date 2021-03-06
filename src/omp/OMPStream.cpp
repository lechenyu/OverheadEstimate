
// Copyright (c) 2015-16 Tom Deakin, Simon McIntosh-Smith,
// University of Bristol HPC
//
// For full license terms please see the LICENSE file distributed with this
// source code

#include <cstdlib>  // For aligned_alloc
#include <cassert>
#include "OMPStream.h"

#ifndef ALIGNMENT
#define ALIGNMENT (2*1024*1024) // 2MB
#endif

#ifdef ESTIMATE
void verify_shadow(const std::string &name, shadow_type *shadow, shadow_type value, int len) 
{
  for (int i = 0; i < len; i++) 
  {
    if (shadow[i] != value) 
    {
      std::cout << "Some elements in " << name << " are not equal to " << value << std::endl;
      std::cout << "The first inconsistent element is " << name << "[" << i << "] = " << shadow[i] << std::endl;
      assert(0);
    }
  }
  std::cout << name << " passes the verification" << std::endl;
}

void verify_compressed(const std::string &name, uint8_t *compressed, shadow_type value, int len, int offset)
{
  for (int i = 0; i < len; i++)
  {
    int val = compressed[offset + i / 2];
    if ((i & 0x00000001) == 0) 
    {
      val = val & 0x00000007;
    }
    else
    {
      val = (val >> 4) & 0x00000007;
    }

    if (val != value) 
    {
      std::cout << "Some elements in " << name << " are not equal to " << value << std::endl;
      std::cout << "The first inconsistent element is " << compressed << "[" << offset + i / 2 << "] = " << compressed[offset + i / 2] << std::endl;
      assert(0);
    }
  }
  std::cout << name << " passes the verification after compression" << std::endl;
} 
#endif

template <class T>
OMPStream<T>::OMPStream(const int ARRAY_SIZE, int device)
{
  array_size = ARRAY_SIZE;

  // Allocate on the host
  this->a = (T*)aligned_alloc(ALIGNMENT, sizeof(T)*array_size);
  this->b = (T*)aligned_alloc(ALIGNMENT, sizeof(T)*array_size);
  this->c = (T*)aligned_alloc(ALIGNMENT, sizeof(T)*array_size);

#ifdef ESTIMATE
  this->sa = new shadow_type[array_size]{0};
  this->sb = new shadow_type[array_size]{0};
  this->sc = new shadow_type[array_size]{0};
  unsigned compressed_array_size = (array_size + 1) / 2;
  this->compressed = new uint8_t[compressed_array_size * 3]{0};

  shadow_type *sa = this->sa;
  shadow_type *sb = this->sb;
  shadow_type *sc = this->sc;
  uint8_t *compressed = this->compressed;
#endif

#ifdef OMP_TARGET_GPU
  omp_set_default_device(device);
  T *a = this->a;
  T *b = this->b;
  T *c = this->c;

  // Set up data region on device
  #pragma omp target enter data map(alloc: a[0:array_size], b[0:array_size], c[0:array_size]) \
                                shadow_mem(alloc, sa, array_size) \
                                shadow_mem(alloc, sb, array_size) \
                                shadow_mem(alloc, sc, array_size) \
                                shadow_mem(alloc, compressed, compressed_array_size * 3)
  {}

#ifdef ESTIMATE
  // #pragma omp target parallel
  // {
    init_shadow(sa, array_size, 0)
    init_shadow(sb, array_size, 0)
    init_shadow(sc, array_size, 0)
  // }
#endif

#endif

#ifdef ESTIMATE
  std::cout << "Estimate Version\n";
#else
  std::cout << "Original Version\n";
#endif
}

template <class T>
OMPStream<T>::~OMPStream()
{
#ifdef OMP_TARGET_GPU
  // End data region on device
  int array_size = this->array_size;
  T *a = this->a;
  T *b = this->b;
  T *c = this->c;
  #pragma omp target exit data map(release: a[0:array_size], b[0:array_size], c[0:array_size]) shadow_mem(release, sa, array_size) shadow_mem(release, sb, array_size) shadow_mem(release, sc, array_size)
  {}
#endif
  free(a);
  free(b);
  free(c);

#ifdef ESTIMATE
  delete[] sa;
  delete[] sb;
  delete[] sc;
#endif
}

template <class T>
void OMPStream<T>::init_arrays(T initA, T initB, T initC)
{
  int array_size = this->array_size;
#ifdef OMP_TARGET_GPU
  T *a = this->a;
  T *b = this->b;
  T *c = this->c;

#ifdef ESTIMATE
  shadow_type *sa = this->sa;
  shadow_type *sb = this->sb;
  shadow_type *sc = this->sc;
  // #pragma omp target parallel
  // {
    init_shadow(sa, array_size, 0)
    init_shadow(sb, array_size, 0)
    init_shadow(sc, array_size, 0)
  // }
#endif

  #pragma omp target teams distribute parallel for simd
  for (int i = 0; i < array_size; i++)
  {
    a[i] = initA;
    b[i] = initB;
    c[i] = initC;
#ifdef ESTIMATE    
    record_w(sa, i)
    record_w(sb, i)
    record_w(sc, i)
#endif
  }

#ifdef ESTIMATE
  #pragma omp target update from(sa[0: array_size], sb[0: array_size], sc[0: array_size])
#endif

#if defined(ESTIMATE) && defined(VERIFY)
  verify_shadow("sa", sa, 0x00000002, array_size);
  verify_shadow("sb", sa, 0x00000002, array_size);
  verify_shadow("sc", sa, 0x00000002, array_size);
#endif

#else
  #pragma omp parallel for
  for (int i = 0; i < array_size; i++)
  {
    a[i] = initA;
    b[i] = initB;
    c[i] = initC;
  }
#endif

  #if defined(OMP_TARGET_GPU) && defined(_CRAYC)
  // If using the Cray compiler, the kernels do not block, so this update forces
  // a small copy to ensure blocking so that timing is correct
  #pragma omp target update from(a[0:0])
  #endif
}

template <class T>
void OMPStream<T>::read_arrays(std::vector<T>& h_a, std::vector<T>& h_b, std::vector<T>& h_c)
{

#ifdef OMP_TARGET_GPU
  T *a = this->a;
  T *b = this->b;
  T *c = this->c;
  #pragma omp target update from(a[0:array_size], b[0:array_size], c[0:array_size])
  {}
#endif

  #pragma omp parallel for simd
  for (int i = 0; i < array_size; i++)
  {
    h_a[i] = a[i];
    h_b[i] = b[i];
    h_c[i] = c[i];
  }

}

template <class T>
void OMPStream<T>::copy()
{
#ifdef OMP_TARGET_GPU
  int array_size = this->array_size;
  T *a = this->a;
  T *c = this->c;
#ifdef ESTIMATE
  shadow_type *sa = this->sa;
  shadow_type *sc = this->sc;

  // #pragma omp target parallel
  // {
    init_shadow(sa, array_size, 0)
    init_shadow(sc, array_size, 0)
  // }
#endif

  #pragma omp target teams distribute parallel for simd
  for (int i = 0; i < array_size; i++)
  {
    c[i] = a[i];

#ifdef ESTIMATE
    record_r(sa, i)
    record_w(sc, i)
#endif  
  }
  
#ifdef ESTIMATE
  #pragma omp target update from(sa[0: array_size], sc[0: array_size])
#endif

#if defined(ESTIMATE) && defined(VERIFY)
  verify_shadow("sa", sa, 0x00000005, array_size);
  verify_shadow("sc", sc, 0x00000002, array_size);
#endif

#else
  #pragma omp parallel for
  for (int i = 0; i < array_size; i++)
  {
    c[i] = a[i];  
  }
#endif

  #if defined(OMP_TARGET_GPU) && defined(_CRAYC)
  // If using the Cray compiler, the kernels do not block, so this update forces
  // a small copy to ensure blocking so that timing is correct
  #pragma omp target update from(a[0:0])
  #endif
}

template <class T>
void OMPStream<T>::mul()
{
  const T scalar = startScalar;

#ifdef OMP_TARGET_GPU
  int array_size = this->array_size;
  T *b = this->b;
  T *c = this->c;
#ifdef ESTIMATE
  shadow_type *sb = this->sb;
  shadow_type *sc = this->sc;
  // #pragma omp target parallel
  // {
    init_shadow(sb, array_size, 0)
    init_shadow(sc, array_size, 0)
  // }
#endif

  #pragma omp target teams distribute parallel for simd
  for (int i = 0; i < array_size; i++)
  {
    b[i] = scalar * c[i];

#ifdef ESTIMATE
    record_r(sc, i)
    record_w(sb, i)
#endif   
  }

#ifdef ESTIMATE
  #pragma omp target update from(sb[0: array_size], sc[0: array_size])
#endif

#if defined(ESTIMATE) && defined(VERIFY)
  verify_shadow("sb", sb, 0x00000002, array_size);
  verify_shadow("sc", sc, 0x00000005, array_size);
#endif

#else
  #pragma omp parallel for
  for (int i = 0; i < array_size; i++)
  {
    b[i] = scalar * c[i];   
  }
#endif

  #if defined(OMP_TARGET_GPU) && defined(_CRAYC)
  // If using the Cray compiler, the kernels do not block, so this update forces
  // a small copy to ensure blocking so that timing is correct
  #pragma omp target update from(c[0:0])
  #endif
}

template <class T>
void OMPStream<T>::add()
{
#ifdef OMP_TARGET_GPU
  int array_size = this->array_size;
  T *a = this->a;
  T *b = this->b;
  T *c = this->c;

#ifdef ESTIMATE
  shadow_type *sa = this->sa;
  shadow_type *sb = this->sb;
  shadow_type *sc = this->sc;
  // #pragma omp target parallel
  // {
    init_shadow(sa, array_size, 0)
    init_shadow(sb, array_size, 0)
    init_shadow(sc, array_size, 0)
  // }
#endif

  #pragma omp target teams distribute parallel for simd
  for (int i = 0; i < array_size; i++)
  {
    c[i] = a[i] + b[i];

#ifdef ESTIMATE
    record_r(sa, i)
    record_r(sb, i)
    record_w(sc, i)
#endif
  }

#ifdef ESTIMATE
  #pragma omp target update from(sa[0: array_size], sb[0: array_size], sc[0: array_size])
#endif

#if defined(ESTIMATE) && defined(VERIFY)
  verify_shadow("sa", sa, 0x00000005, array_size);
  verify_shadow("sb", sb, 0x00000005, array_size);
  verify_shadow("sc", sc, 0x00000002, array_size);
#endif

#else
  #pragma omp parallel for
  for (int i = 0; i < array_size; i++)
  {
    c[i] = a[i] + b[i];
  }
#endif

  #if defined(OMP_TARGET_GPU) && defined(_CRAYC)
  // If using the Cray compiler, the kernels do not block, so this update forces
  // a small copy to ensure blocking so that timing is correct
  #pragma omp target update from(a[0:0])
  #endif
}

template <class T>
void OMPStream<T>::triad()
{
  const T scalar = startScalar;

#ifdef OMP_TARGET_GPU
  int array_size = this->array_size;
  T *a = this->a;
  T *b = this->b;
  T *c = this->c;

#ifdef ESTIMATE
  shadow_type *sa = this->sa;
  shadow_type *sb = this->sb;
  shadow_type *sc = this->sc;
  // #pragma omp target parallel
  // {
    init_shadow(sa, array_size, 0)
    init_shadow(sb, array_size, 0)
    init_shadow(sc, array_size, 0)
  // }
#endif

  #pragma omp target teams distribute parallel for simd
  for (int i = 0; i < array_size; i++)
  {
    a[i] = b[i] + scalar * c[i];

#ifdef ESTIMATE
    // #pragma omp atomic update
    record_r(sb, i)
    // #pragma omp atomic update
    record_r(sc, i)
    // #pragma omp atomic update
    record_w(sa, i)
#endif
  }

#ifdef ESTIMATE
  
#ifdef OP1
  // we assume each shadow array's size is always an even numer
  unsigned compressed_array_size = (array_size + 1) / 2;
  //uint8_t *compressed = new uint8_t[compressed_array_size * 3]{0};
  uint8_t *compressed = this->compressed;
  unsigned offset = 0;
  // #pragma omp target data map(from: compressed[0: compressed_array_size * 3]) // 800GB/s 239.2GB/s 45.7GB/s 29.9GB/s 
  // {
  // try a different partition: each thread write to a word (4 bytes)
  #pragma omp target teams distribute parallel for
  for (int i = 0; i < array_size; i += 2)
  {
    int idx1 = i, idx2 = i + 1;
    compressed[offset + idx1 / 2] = ((sa[idx1] & 0x00000007) << 4) | (sa[idx2] & 0x00000007);
  }
  offset += compressed_array_size;

  #pragma omp target teams distribute parallel for
  for (int i = 0; i < array_size; i += 2)
  {
    int idx1 = i, idx2 = i + 1;
    compressed[offset + idx1 / 2] = ((sb[idx1] & 0x00000007) << 4) | (sb[idx2] & 0x00000007);
  }
  offset += compressed_array_size;

  #pragma omp target teams distribute parallel for
  for (int i = 0; i < array_size; i += 2)
  {
    int idx1 = i, idx2 = i + 1;
    compressed[offset + idx1 / 2] = ((sc[idx1] & 0x00000007) << 4) | (sc[idx2] & 0x00000007);
  }

  // }
  #pragma omp target update from(compressed[0: compressed_array_size * 3])
#ifdef VERIFY
  verify_compressed("sa", compressed, 0x00000002, array_size, 0);
  verify_compressed("sb", compressed, 0x00000005, array_size, compressed_array_size);
  verify_compressed("sc", compressed, 0x00000005, array_size, compressed_array_size * 2);
#endif

#elif defined(OP2)

#else
  #pragma omp target update from(sa[0: array_size])
  #pragma omp target update from(sb[0: array_size])
  #pragma omp target update from(sc[0: array_size])

#ifdef VERIFY
  verify_shadow("sa", sa, 0x00000002, array_size);
  verify_shadow("sb", sb, 0x00000005, array_size);
  verify_shadow("sc", sc, 0x00000005, array_size);
#endif
#endif // OP
#endif // ESTIMATE



#else
  #pragma omp parallel for
  for (int i = 0; i < array_size; i++)
  {
    a[i] = b[i] + scalar * c[i];
  }
#endif

  #if defined(OMP_TARGET_GPU) && defined(_CRAYC)
  // If using the Cray compiler, the kernels do not block, so this update forces
  // a small copy to ensure blocking so that timing is correct
  #pragma omp target update from(a[0:0])
  #endif
}

template <class T>
void OMPStream<T>::nstream()
{
  const T scalar = startScalar;

#ifdef OMP_TARGET_GPU
  int array_size = this->array_size;
  T *a = this->a;
  T *b = this->b;
  T *c = this->c;

#ifdef ESTIMATE
  shadow_type *sa = this->sa;
  shadow_type *sb = this->sb;
  shadow_type *sc = this->sc;
  // #pragma omp target parallel
  // {
    init_shadow(sa, array_size, 0)
    init_shadow(sb, array_size, 0)
    init_shadow(sc, array_size, 0)
  // }
#endif

  #pragma omp target teams distribute parallel for simd
  for (int i = 0; i < array_size; i++)
  {
    a[i] += b[i] + scalar * c[i];

#ifdef ESTIMATE
    record_r(sb, i)
    record_r(sc, i)
    record_w(sa, i)
#endif
  }

#ifdef ESTIMATE
  #pragma omp target update from(sa[0: array_size], sb[0: array_size], sc[0: array_size])
#endif

#if defined(ESTIMATE) && defined(VERIFY)
  verify_shadow("sa", sa, 0x00000002, array_size);
  verify_shadow("sb", sb, 0x00000005, array_size);
  verify_shadow("sc", sc, 0x00000005, array_size);
#endif

#else
  #pragma omp parallel for
  for (int i = 0; i < array_size; i++)
  {
    a[i] += b[i] + scalar * c[i];
  }
#endif

  #if defined(OMP_TARGET_GPU) && defined(_CRAYC)
  // If using the Cray compiler, the kernels do not block, so this update forces
  // a small copy to ensure blocking so that timing is correct
  #pragma omp target update from(a[0:0])
  #endif
}

template <class T>
T OMPStream<T>::dot()
{
  T sum = 0.0;

#ifdef OMP_TARGET_GPU
  int array_size = this->array_size;
  T *a = this->a;
  T *b = this->b;

#ifdef ESTIMATE
  shadow_type *sa = this->sa;
  shadow_type *sb = this->sb;
  // #pragma omp target parallel
  // {
    init_shadow(sa, array_size, 0)
    init_shadow(sb, array_size, 0)
  // }
#endif

  #pragma omp target teams distribute parallel for simd map(tofrom: sum) reduction(+:sum)
  for (int i = 0; i < array_size; i++)
  {
    sum += a[i] * b[i];

#ifdef ESTIMATE
    record_r(sa, i)
    record_r(sb, i)
#endif
  }

#ifdef ESTIMATE
  #pragma omp target update from(sa[0: array_size], sb[0: array_size])
#endif

#if defined(ESTIMATE) && defined(VERIFY)
  verify_shadow("sa", sa, 0x00000005, array_size);
  verify_shadow("sb", sb, 0x00000005, array_size);
#endif

#else
  #pragma omp parallel for reduction(+:sum)
  for (int i = 0; i < array_size; i++)
  {
    sum += a[i] * b[i];
  }
#endif

  return sum;
}



void listDevices(void)
{
#ifdef OMP_TARGET_GPU
  // Get number of devices
  int count = omp_get_num_devices();

  // Print device list
  if (count == 0)
  {
    std::cerr << "No devices found." << std::endl;
  }
  else
  {
    std::cout << "There are " << count << " devices." << std::endl;
  }
#else
  std::cout << "0: CPU" << std::endl;
#endif
}

std::string getDeviceName(const int)
{
  return std::string("Device name unavailable");
}

std::string getDeviceDriver(const int)
{
  return std::string("Device driver unavailable");
}
template class OMPStream<float>;
template class OMPStream<double>;
