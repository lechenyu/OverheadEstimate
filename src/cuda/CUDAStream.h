
// Copyright (c) 2015-16 Tom Deakin, Simon McIntosh-Smith,
// University of Bristol HPC
//
// For full license terms please see the LICENSE file distributed with this
// source code

#pragma once

#include <iostream>
#include <stdexcept>
#include <sstream>

#include "Stream.h"

#if defined(PAGEFAULT)
  #define IMPLEMENTATION_STRING "CUDA - Page Fault"
#elif defined(MANAGED)
  #define IMPLEMENTATION_STRING "CUDA - Managed Memory"
#else
  #define IMPLEMENTATION_STRING "CUDA"
#endif

#define TBSIZE 1024
#define DOT_NUM_BLOCKS 256

#define ESTIMATE 1

#ifdef ESTIMATE
#define record_r(x, i) x[i] |= ((~x[i] & 0x00000002) >> 1);
#define record_w(x, i) x[i] |= 0x00000002;
// #else
// #define record_r(x, i)
// #define record_w(x, i)
#endif

template <class T>
class CUDAStream : public Stream<T>
{
  protected:
    // Size of arrays
    int array_size;

    // Host array for partial sums for dot kernel
    T *sums;

    // Device side pointers to arrays
    T *d_a;
    T *d_b;
    T *d_c;
    T *d_sum;

#ifdef ESTIMATE
    uint32_t *sa;
    uint32_t *sb;
    uint32_t *sc;
    uint32_t *ssum;
#endif

  public:

    CUDAStream(const int, const int);
    ~CUDAStream();

    virtual void copy() override;
    virtual void add() override;
    virtual void mul() override;
    virtual void triad() override;
    virtual void nstream() override;
    virtual T dot() override;

    virtual void init_arrays(T initA, T initB, T initC) override;
    virtual void read_arrays(std::vector<T>& a, std::vector<T>& b, std::vector<T>& c) override;

};
