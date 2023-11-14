#include <unistd.h>

#define NSYS_ATTACH                                                            \
  "\
#/bin/bash \n\
nsys start -o gpudb-perf --session gpudb-perf -f true\
"

#define NSYS_DETACH                                                            \
  "\
#/bin/bash \n\
nsys stop --session gpudb-perf\
"

void nsys_attach(int trial, int num_trial) {
  // if (trial == num_trial) {
  //   system(NSYS_ATTACH);
  //   usleep(1000 * 1000);
  // }
}

void nsys_detach(int trial, int num_trial) {
  // if (trial == num_trial) {
  //   usleep(1000 * 1000);
  //   system(NSYS_DETACH);
  // }
}