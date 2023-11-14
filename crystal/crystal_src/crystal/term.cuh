#pragma once

template <typename T, int BLOCK_THREADS, int ITEMS_PER_THREAD>
__device__ __forceinline__ void
Term(int (&selection_flags)[ITEMS_PER_THREAD]) {
    int count = 0;
    for (int ITEM = 0; ITEM < ITEMS_PER_THREAD; ITEM++) {
        count += selection_flags[ITEM];
    }
    if (count == 0) {
        printf("Term\n");
        return;
    }
}