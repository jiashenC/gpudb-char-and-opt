// Ensure printing of CUDA runtime errors to console
#define CUB_STDERR

#include <chrono>
#include <curand.h>
#include <fstream>
#include <iostream>
#include <stdio.h>
#include <string>
#include <unistd.h>

#include "cub/test/test_util.h"
#include <cub/util_allocator.cuh>
#include <cuda.h>

#include "crystal/crystal.cuh"

#include "utils/generator.h"
#include "utils/gpu_utils.h"

using namespace std;

#define DEBUG 1

template <int BLOCK_THREADS, int ITEMS_PER_THREAD>
__global__ void aggregate_kernel(int *key, int *val, int num_tuples,
                                 int *hash_table, int num_slots) {
  int items[ITEMS_PER_THREAD];
  int items2[ITEMS_PER_THREAD];

  int tile_offset = blockIdx.x * TILE_SIZE;
  int num_tiles = (num_tuples + TILE_SIZE - 1) / TILE_SIZE;
  int num_tile_items = TILE_SIZE;

  if (blockIdx.x == num_tiles - 1) {
    num_tile_items = num_tuples - tile_offset;
  }

  BlockLoad<int, BLOCK_THREADS, ITEMS_PER_THREAD>(key + tile_offset, items,
                                                  num_tile_items);
  BlockLoad<int, BLOCK_THREADS, ITEMS_PER_THREAD>(val + tile_offset, items2,
                                                  num_tile_items);

  for (int ITEM = 0; ITEM < ITEMS_PER_THREAD; ITEM++) {
    int key = items[ITEM], val = items2[ITEM];
    hash_table[key * 2] = key;
    atomicAdd(&hash_table[key * 2 + 1], val);
  }
}

struct TimeKeeper {
  float time_build;
  float time_probe;
  float time_extra;
  float time_total;
};

TimeKeeper Aggregate(int *d_key, int *d_val, int num_rows, int num_slots,
                     cub::CachingDeviceAllocator &g_allocator) {
  SETUP_TIMING();

  int *hash_table = NULL;
  float time_build, time_probe, time_memset, time_memset2;

  ALLOCATE(hash_table, sizeof(int) * 2 * num_slots);
  cudaMemset(hash_table, 0, num_slots * sizeof(int) * 2);

  int tile_items = 128 * 4;

  aggregate_kernel<128, 4><<<(num_rows + tile_items - 1) / tile_items, 128>>>(
      d_key, d_val, num_rows, hash_table, num_slots);
  cudaDeviceSynchronize();

#if DEBUG
  /* cout << "{" */
  /*      << "\"time_memset\":" << time_memset << ",\"time_build\"" <<
   * time_build */
  /*      << ",\"time_probe\":" << time_probe << "}" << endl; */
#endif

  CLEANUP(hash_table);

  TimeKeeper t = {time_build, time_probe, time_memset,
                  time_build + time_probe + time_memset};
  return t;
}

//---------------------------------------------------------------------
// Globals, constants and typedefs
//---------------------------------------------------------------------

bool g_verbose = false; // Whether to display input/output to console
cub::CachingDeviceAllocator
    g_allocator(true); // Caching allocator for device memory

#define CLEANUP(vec)                                                           \
  if (vec)                                                                     \
  CubDebugExit(g_allocator.DeviceFree(vec))

//---------------------------------------------------------------------
// Main
//---------------------------------------------------------------------
int main(int argc, char **argv) {
  using milli = chrono::milliseconds;
  auto st = chrono::high_resolution_clock::now();

  int num_rows = 268435456;
  int num_slots = 100;
  int num_trials = 3;

  // Initialize command line
  CommandLineArgs args(argc, argv);
  args.GetCmdLineArgument("r", num_rows);
  args.GetCmdLineArgument("s", num_slots);
  args.GetCmdLineArgument("t", num_trials);

  // Print usage
  if (args.CheckCmdLineFlag("help")) {
    printf("%s "
           "[--r=<num rows>] "
           "[--s=<num slots>] "
           "[--t=<num trials>] "
           "[--device=<device-id>] "
           "[--v] "
           "\n",
           argv[0]);
    exit(0);
  }

  // Initialize device
  CubDebugExit(args.DeviceInit());

  int *h_key = (int *)malloc(sizeof(int) * num_rows);
  int *h_val = (int *)malloc(sizeof(int) * num_rows);

  ifstream if_key("./data/storage/aggregate_pk_" + to_string(num_rows) + "_" +
                      to_string(num_slots) + ".bin",
                  ios::in | ios::binary);
  if (!if_key.good()) {
    cout << "./data/storage/aggregate_pk_" + to_string(num_rows) + "_" +
                to_string(num_slots) + ".bin does not exit"
         << endl;
    exit(1);
  }
  if_key.read(reinterpret_cast<char *>(h_key), sizeof(int) * num_rows);
  if_key.close();

  ifstream if_val("./data/storage/aggregate_attr_" + to_string(num_rows) + "_" +
                      to_string(num_slots) + ".bin",
                  ios::in | ios::binary);
  if (!if_val.good()) {
    cout << "./data/storage/aggregate_attr_" + to_string(num_rows) + "_" +
                to_string(num_slots) + ".bin does not exit"
         << endl;
    exit(1);
  }
  if_val.read(reinterpret_cast<char *>(h_val), sizeof(int) * num_rows);
  if_val.close();

  // create_relation_pk(h_dim_key, h_dim_val, num_dim);
  // create_relation_fk(h_fact_fkey, h_fact_val, num_fact, num_dim);

  // trial counting
  int trial = 1;

  nsys_attach(trial, num_trials);

  // Allocate problem device arrays
  int *d_key = NULL;
  int *d_val = NULL;
  CubDebugExit(
      g_allocator.DeviceAllocate((void **)&d_key, sizeof(int) * num_rows));
  CubDebugExit(
      g_allocator.DeviceAllocate((void **)&d_val, sizeof(int) * num_rows));

  CubDebugExit(
      cudaMemcpy(d_key, h_key, sizeof(int) * num_rows, cudaMemcpyHostToDevice));
  CubDebugExit(
      cudaMemcpy(d_val, h_val, sizeof(int) * num_rows, cudaMemcpyHostToDevice));

  Aggregate(d_key, d_val, num_rows, num_slots, g_allocator);

  auto fin = chrono::high_resolution_clock::now();
  cout << chrono::duration_cast<milli>(fin - st).count() << "ms" << endl;

  nsys_detach(trial, num_trials);

  for (int j = 0; j < num_trials - 1; j++) {
    trial++;

    nsys_attach(trial, num_trials);
    st = chrono::high_resolution_clock::now();
    Aggregate(d_key, d_val, num_rows, num_slots, g_allocator);
    fin = chrono::high_resolution_clock::now();
    cout << chrono::duration_cast<milli>(fin - st).count() << "ms" << endl;
    nsys_detach(trial, num_trials);
  }

  return 0;
}
