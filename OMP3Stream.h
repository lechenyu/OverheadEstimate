
#pragma once

#include <iostream>
#include <stdexcept>

#include "Stream.h"

#define IMPLEMENTATION_STRING "Reference OpenMP"

template <class T>
class OMP3Stream : public Stream<T>
{
  protected:
    // Size of arrays
    unsigned int array_size;
    // Device side pointers
    T *a;
    T *b;
    T *c;

  public:
    OMP3Stream(const unsigned int, T*, T*, T*);
    ~OMP3Stream();

    virtual void copy() override;
    virtual void add() override;
    virtual void mul() override;
    virtual void triad() override;

    virtual void write_arrays(const std::vector<T>& a, const std::vector<T>& b, const std::vector<T>& c) override;
    virtual void read_arrays(std::vector<T>& a, std::vector<T>& b, std::vector<T>& c) override;

};

