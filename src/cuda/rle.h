#ifndef RLE_H
#define RLE_H

#include <iostream>
#include <string>
#include <chrono>

const int BLOCK_SIZE = 1024;
const int BLK_LEN_LIMIT = 2 * BLOCK_SIZE;
const int COLUMN_SIZE = 10;

template <typename T> class RleResult {
public:
  T *value;
  int *index;
  size_t len;
  bool is_on_heap;
  bool is_on_device;

  RleResult() = delete;
  RleResult(T *value, int *index, size_t len, bool is_on_heap = true, bool is_on_device = true)
      : value(value), index(index), len(len), is_on_heap(is_on_heap), is_on_device(is_on_device){}
  ~RleResult() {
    //std::cout << "RleResult released" << std::endl;
    if (is_on_heap) {
      if (is_on_device) {
        cudaFree(value);
        cudaFree(index);
      } else {
        delete[] value;
        delete[] index;
      }
    }
  }
};

template<typename T>
void print_array(const std::string &name, T array, size_t len) {
  std::cout << name << ":\n";
  int columns = (len + COLUMN_SIZE - 1) / COLUMN_SIZE;
  int i;
  for (i = 0; i < columns; i++) {
    int bound;
    if (i < columns - 1) {
      bound = COLUMN_SIZE;
    } else {
      bound = len - (columns - 1) * COLUMN_SIZE;
    }
    std::cout << array[i * COLUMN_SIZE];
    for (int j = 1; j < bound; j++) {
      std::cout << " " << array[i * COLUMN_SIZE + j];
    }
    std::cout << "\n";
  }
}

// size shall be twice BLOCK_SIZE
template<typename T>
__global__ void scan(T *d_input, T *d_output, T *d_partial_sum) {
  const size_t size = BLK_LEN_LIMIT;
  __shared__ T temp[size];
  int tid = threadIdx.x;
  T *blk_input = d_input + blockIdx.x * size;
  T *blk_output = d_output + blockIdx.x * size;
  temp[tid * 2] = blk_input[tid * 2];
  temp[tid * 2 + 1] = blk_input[tid * 2 + 1];
  int sec = 1;
  for (int d = size >> 1; d > 0; d >>= 1) {
    if (tid < d) {  
      int right = (2 * tid + 2) * sec - 1;
      int left = right - sec;
      temp[right] += temp[left];
    }
    sec <<= 1;
    __syncthreads();
  }
  if (tid == 0) {
    if (d_partial_sum) {
      d_partial_sum[blockIdx.x] = temp[size - 1];
    }
    blk_output[size - 1] = temp[size - 1];
    temp[size - 1] = 0;
  }
  sec = size >> 1;
  for (int d = 1; d < size; d <<= 1) {
    __syncthreads();
    if (tid < d) {
      int right = (2 * tid + 2) * sec - 1;
      int left = right - sec;
      T prev_right = temp[right];
      temp[right] += temp[left];
      temp[left] = prev_right;
    }
    sec >>= 1;
  }
  if (tid > 0) {
    int left = 2 * tid;
    int right = 2 * tid + 1;
    blk_output[left - 1] = temp[left];
    blk_output[right - 1] = temp[right]; 
  } else {
    blk_output[0] = temp[1];
  }
}

template<typename T>
__global__ void add_partial_sum(T *d_output, T *d_partial_sum) {
  // int pos1 = (blockIdx.x + 1) * blockDim.x + threadIdx.x;
  // output[2 * pos1] += partial_sum[blockIdx.x];
  // output[2 * pos1 + 1] += partial_sum[blockIdx.x];

  // int pos1 = blockIdx.x * blockDim.x + threadIdx.x;
  // int pos2 = pos1 + gridDim.x * blockDim.x;
  // output[pos1 + 2 * blockDim.x] += partial_sum[blockIdx.x >> 1];
  // output[pos2 + 2 * blockDim.x] += partial_sum[(blockIdx.x + gridDim.x) >> 1];

  int pos = blockIdx.x * blockDim.x + threadIdx.x;
  d_output[pos] += d_partial_sum[blockIdx.x >> 1];
}

template<typename T>
void prefix_sum_impl(T *d_input, T *d_output, int block_num) {
  if (block_num == 1) {
    scan<<<block_num, BLOCK_SIZE>>>(d_input, d_output, (T *)nullptr);
  } else {
    T *d_partial_sum;
    int block_num2 = (block_num + BLK_LEN_LIMIT - 1) / BLK_LEN_LIMIT;
    size_t partial_sum_len = block_num2 * BLK_LEN_LIMIT;
    size_t partial_sum_size = partial_sum_len * sizeof(T);
    cudaMalloc(&d_partial_sum, partial_sum_size);
    scan<<<block_num, BLOCK_SIZE>>>(d_input, d_output, d_partial_sum);
    // T *hp = new T[block_num]{};
    // cudaMemcpy(hp, partial_sum, block_num * sizeof(T), cudaMemcpyDeviceToHost);
    // print_array("hp (in)", hp, block_num);
    prefix_sum_impl(d_partial_sum, d_partial_sum, block_num2);
    // cudaMemcpy(hp, partial_sum, block_num * sizeof(T), cudaMemcpyDeviceToHost);
    // print_array("hp (out)", hp, block_num);
    // add_partial_sum<<<block_num - 1, BLOCK_SIZE>>>(output, partial_sum);
    add_partial_sum<<<2 * (block_num - 1), BLOCK_SIZE>>>(d_output + BLK_LEN_LIMIT, d_partial_sum);
    cudaFree(d_partial_sum);
  }
}

template<typename T>
void parallel_prefix_sum(T *input, T *output, size_t len) {
#ifdef PROFILE
  auto before = std::chrono::high_resolution_clock::now();
#endif

  T *d_input, *d_output;
  int block_num = (len + BLK_LEN_LIMIT - 1) / BLK_LEN_LIMIT;
  size_t padded_len = block_num * BLK_LEN_LIMIT;
  size_t size = len * sizeof(T);
  size_t padded_size = padded_len * sizeof(T);
  cudaMalloc(&d_input, padded_size * 2);
  cudaMemset(d_input, 0, padded_size * 2);
  d_output = d_input + padded_len;
  cudaMemcpy(d_input, input, size, cudaMemcpyHostToDevice);

#ifdef PROFILE
  auto before2 = std::chrono::high_resolution_clock::now();
#endif

  prefix_sum_impl(d_input, d_output, block_num);

#ifdef PROFILE
  std::chrono::duration<double> elapse2 = std::chrono::high_resolution_clock::now() - before2;
  std::cout << "Prefix sum (len = " << len << ") kernel execution time: " << elapse2.count() << " seconds" << std::endl;
#endif

  cudaMemcpy(output, d_output, size, cudaMemcpyDeviceToHost);
  cudaFree(d_input);

#ifdef PROFILE
  std::chrono::duration<double> elapse = std::chrono::high_resolution_clock::now() - before;
  std::cout << "Prefix sum (len = " << len << ") overall execution time: " << elapse.count() << " seconds" << std::endl;
#endif
}

template<typename T>
__global__ void get_mask(T *d_input, int *d_output, size_t len) {
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  if (tid == 0) {
    d_output[tid] = 1;
  } else if (tid < len) {
    d_output[tid] = (d_input[tid] != d_input[tid - 1]);
  }
}

__global__ void get_compact_mask(int *d_mask, int *d_compact_mask, size_t len) {
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  if (tid == len - 1) {
    d_compact_mask[0] = 0;
    d_compact_mask[d_mask[tid]] = len;
  } else if (tid < len -1 && d_mask[tid + 1] > d_mask[tid]) {
    d_compact_mask[d_mask[tid + 1] - 1] = tid + 1;
  }
}

template<typename T>
__global__ void get_rle_result(T *d_input, int *d_compact_mask, int *d_index, T *d_value, size_t len) {
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  if (tid < len) {
    d_value[tid] = d_input[d_compact_mask[tid]];
    d_index[tid] = d_compact_mask[tid + 1] - d_compact_mask[tid];
  }
}

template<typename T>
RleResult<T> parallel_rle_impl(T *d_input, size_t len) {
  T *d_value;
  int *d_mask, *d_compact_mask, *d_index;
  int block_num = (len + BLOCK_SIZE - 1) / BLOCK_SIZE;
  int block_num2 = (len + BLK_LEN_LIMIT - 1) / BLK_LEN_LIMIT;
  cudaMalloc(&d_mask, block_num2 * BLK_LEN_LIMIT * sizeof(int));

#ifdef PROFILE   
  auto before2 = std::chrono::high_resolution_clock::now();
#endif 

  get_mask<<<block_num, BLOCK_SIZE>>>(d_input, d_mask, len);
  prefix_sum_impl(d_mask, d_mask, block_num2);

#ifdef PROFILE 
  std::chrono::duration<double> elapse2 = std::chrono::high_resolution_clock::now() - before2;
#endif

  int temp;
  cudaMemcpy(&temp, &d_mask[len - 1], sizeof(int), cudaMemcpyDeviceToHost);
  size_t compressed_len = temp;
  //std::cout << "Length after compression: " << compressed_len << std::endl;
  cudaMalloc(&d_compact_mask, (compressed_len + 1) * sizeof(int));
  cudaMalloc(&d_value, compressed_len * sizeof(T));
  cudaMalloc(&d_index, compressed_len * sizeof(int));

#ifdef PROFILE 
  before2 = std::chrono::high_resolution_clock::now();
#endif

  get_compact_mask<<<block_num, BLOCK_SIZE>>>(d_mask, d_compact_mask, len);
  int block_num3 = (compressed_len + BLOCK_SIZE - 1) / BLOCK_SIZE;
  get_rle_result<<<block_num3, BLOCK_SIZE>>>(d_input, d_compact_mask, d_index, d_value, compressed_len);

#ifdef PROFILE 
  elapse2 += (std::chrono::high_resolution_clock::now() - before2);
  std::cout << "Run length encoding (len = " << len << ") kernel execution time: " << elapse2.count() << " seconds" << std::endl;
#endif

  cudaFree(d_mask);
  cudaFree(d_compact_mask);
  return RleResult<T>(d_value, d_index, compressed_len);
}

template<typename T>
RleResult<T> parallel_rle(T *input, size_t len) {
#ifdef PROFILE 
  auto before = std::chrono::high_resolution_clock::now();
#endif

  T *d_input, *value;
  int *index;
  size_t size = len * sizeof(T);
  cudaMalloc(&d_input, size);
  cudaMemcpy(d_input, input, size, cudaMemcpyHostToDevice);
  RleResult<T> &&d_result = parallel_rle_impl(d_input, len);
  size_t compressed_len = d_result.len; 
  value = new T[compressed_len];
  index = new int[compressed_len];
  cudaMemcpy(value, d_result.value, compressed_len * sizeof(T), cudaMemcpyDeviceToHost);
  cudaMemcpy(index, d_result.index, compressed_len * sizeof(int), cudaMemcpyDeviceToHost);
  cudaFree(d_input);

#ifdef PROFILE 
  std::chrono::duration<double> elapse = std::chrono::high_resolution_clock::now() - before;
  std::cout << "Run length encoding (len = " << len << ") overall execution time: " << elapse.count() << " seconds" << std::endl;
#endif
  return RleResult<T>{value, index, compressed_len, true, false};
}
#endif // RLE_H
