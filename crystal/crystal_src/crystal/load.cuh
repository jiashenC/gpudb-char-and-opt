#pragma once

template <typename T, int BLOCK_THREADS, int ITEMS_PER_THREAD>
__device__ __forceinline__ void
BlockPredLoadDirect(const unsigned int tid, T *block_itr,
                    T (&items)[ITEMS_PER_THREAD],
                    int (&selection_flags)[ITEMS_PER_THREAD]) {
  T *thread_itr = block_itr + tid;

#pragma unroll
  for (int ITEM = 0; ITEM < ITEMS_PER_THREAD; ITEM++) {
    if (selection_flags[ITEM]) {
      items[ITEM] = thread_itr[ITEM * BLOCK_THREADS];
    }
  }
}

template <typename T, int BLOCK_THREADS, int ITEMS_PER_THREAD>
__device__ __forceinline__ void
BlockPredLoadDirect(const unsigned int tid, T *block_itr,
                    T (&items)[ITEMS_PER_THREAD],
                    int (&selection_flags)[ITEMS_PER_THREAD], int num_items) {
  T *thread_itr = block_itr + tid;

#pragma unroll
  for (int ITEM = 0; ITEM < ITEMS_PER_THREAD; ITEM++) {
    if (tid + (ITEM * BLOCK_THREADS) < num_items && selection_flags[ITEM]) {
      items[ITEM] = thread_itr[ITEM * BLOCK_THREADS];
    }
  }
}

template <typename T, int BLOCK_THREADS, int ITEMS_PER_THREAD>
__device__ __forceinline__ void
BlockPredLoad(T *inp, T (&items)[ITEMS_PER_THREAD],
              int (&selection_flags)[ITEMS_PER_THREAD], int num_items) {
  T *block_itr = inp;

  if ((BLOCK_THREADS * ITEMS_PER_THREAD) == num_items) {
    BlockPredLoadDirect<T, BLOCK_THREADS, ITEMS_PER_THREAD>(
        threadIdx.x, block_itr, items, selection_flags);
  } else {
    BlockPredLoadDirect<T, BLOCK_THREADS, ITEMS_PER_THREAD>(
        threadIdx.x, block_itr, items, selection_flags, num_items);
  }
}

template <typename T, int BLOCK_THREADS, int ITEMS_PER_THREAD>
__device__ __forceinline__ void BlockLoadDirect(const unsigned int tid,
                                                T *block_itr,
                                                T (&items)[ITEMS_PER_THREAD]) {
  T *thread_itr = block_itr + tid;

#pragma unroll
  for (int ITEM = 0; ITEM < ITEMS_PER_THREAD; ITEM++) {
    items[ITEM] = thread_itr[ITEM * BLOCK_THREADS];
  }
}

template <typename T, int BLOCK_THREADS, int ITEMS_PER_THREAD>
__device__ __forceinline__ void
BlockLoadDirect(const unsigned int tid, T *block_itr,
                T (&items)[ITEMS_PER_THREAD], int num_items) {
  T *thread_itr = block_itr + tid;

#pragma unroll
  for (int ITEM = 0; ITEM < ITEMS_PER_THREAD; ITEM++) {
    if (tid + (ITEM * BLOCK_THREADS) < num_items) {
      items[ITEM] = thread_itr[ITEM * BLOCK_THREADS];
    }
  }
}

template <typename T, int BLOCK_THREADS, int ITEMS_PER_THREAD>
__device__ __forceinline__ void BlockLoad(T *inp, T (&items)[ITEMS_PER_THREAD],
                                          int num_items) {
  T *block_itr = inp;

  if ((BLOCK_THREADS * ITEMS_PER_THREAD) == num_items) {
    BlockLoadDirect<T, BLOCK_THREADS, ITEMS_PER_THREAD>(threadIdx.x, block_itr,
                                                        items);
  } else {
    BlockLoadDirect<T, BLOCK_THREADS, ITEMS_PER_THREAD>(threadIdx.x, block_itr,
                                                        items, num_items);
  }
}

#if 0

template<typename T, int BLOCK_THREADS, int ITEMS_PER_THREAD>
__device__ __forceinline__ void BlockLoadDirect(
    int tid,
    T* block_itr,
    T  (&items)[ITEMS_PER_THREAD]
    ) {
  T* thread_itr = block_itr + tid;

#pragma unroll
  for (int ITEM = 0; ITEM < ITEMS_PER_THREAD; ITEM++) {
    items[ITEM] = thread_itr[ITEM * BLOCK_THREADS];
  }
}

template<typename T, int BLOCK_THREADS, int ITEMS_PER_THREAD>
__device__ __forceinline__ void BlockLoadDirect(
    int tid,
    T* block_itr,
    T  (&items)[ITEMS_PER_THREAD]
    int num_items
    ) {
  T* thread_itr = block_itr + tid;

#pragma unroll
  for (int ITEM = 0; ITEM < ITEMS_PER_THREAD; ITEM++) {
    if (tid + (ITEM * BLOCK_THREADS) < num_items) {
      items[ITEM] = thread_itr[ITEM * BLOCK_THREADS];
    }
  }
}

template<typename T, int BLOCK_THREADS, int ITEMS_PER_THREAD>
__device__ __forceinline__ void BlockLoad(
    T* inp,
    T  (&items)[ITEMS_PER_THREAD]
    int num_items
    ) {
  T* block_itr = inp + blockIdx.x * blockDim.x;

  if ((BLOCK_THREADS * ITEMS_PER_THREAD) == num_items) {
    BlockLoadDirect(threadIdx.x, block_itr, items);
  } else {
    BlockLoadDirect(threadIdx.x, block_itr, items, num_items);
  }
}

#endif
