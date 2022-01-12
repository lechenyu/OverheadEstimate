
// Copyright (c) 2015-16 Tom Deakin, Simon McIntosh-Smith,
// University of Bristol HPC
//
// For full license terms please see the LICENSE file distributed with this
// source code

#pragma once

#include <iostream>
#include <stdexcept>

#include "Stream.h"

#include <omp.h>

#define IMPLEMENTATION_STRING "OpenMP"

#define OMP_TARGET_GPU 1
#define ESTIMATE 1
// #define VERIFY 1

// b0: kernel issues a read before the first write;       0: no, 1: yes
// b1: kernel writes to this location;                    0: no, 1: yes
// b2: kernel reads from this location;                   0: no, 1: yes

#ifdef ESTIMATE
#define shadow_type uint32_t
#define shadow_mem(t, x, s) map(t: x[0:s]) 
#define record_r(x, i) x[i] |= (((~x[i] & 0x00000002) >> 1) | 0x00000004);
#define record_w(x, i) x[i] |= 0x00000002;
#define init_shadow(x, s, v)                     \
  _Pragma("omp for simd")                             \
  for (int i = 0; i < s; i++)                    \
  {                                              \
    x[i] = v;                                    \
  }

#else
#define shadow_mem(t, x, s) 
#define record_r(x, i)
#define record_w(x, i)
#endif

template <class T>
class OMPStream : public Stream<T>
{
  protected:
    // Size of arrays
    int array_size;

    // Device side pointers
    T *a;
    T *b;
    T *c;

#ifdef ESTIMATE
    shadow_type *sa;
    shadow_type *sb;
    shadow_type *sc;
#endif

  public:
    OMPStream(const int, int);
    ~OMPStream();

    virtual void copy() override;
    virtual void add() override;
    virtual void mul() override;
    virtual void triad() override;
    virtual void nstream() override;
    virtual T dot() override;

    virtual void init_arrays(T initA, T initB, T initC) override;
    virtual void read_arrays(std::vector<T>& a, std::vector<T>& b, std::vector<T>& c) override;



};
