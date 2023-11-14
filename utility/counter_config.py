def metric_sol():
    return {
        "sm__throughput.avg.pct_of_peak_sustained_elapsed": "Compute (SM) Throughput",
        "gpu__time_duration.sum": "Duration",
        "gpc__cycles_elapsed.max": "Elapsed Cycles",
        "sm__cycles_active.avg": "SM Active Cycles",
        "gpu__compute_memory_throughput.avg.pct_of_peak_sustained_elapsed": "Memory Throughput",
        "l1tex__throughput.avg.pct_of_peak_sustained_active": "L1/TEX Cache Throughput",
        "lts__throughput.avg.pct_of_peak_sustained_elapsed": "L2 Cache Throughput",
        "gpu__dram_throughput.avg.pct_of_peak_sustained_elapsed": "DRAM Throughput",
    }


def metric_roofline():
    return {
        "sm__sass_thread_inst_executed_op_ffma_pred_on.sum.peak_sustained": "Peak FP32 (FFMA)",
        "sm__sass_thread_inst_executed_op_fmul_pred_on.sum.peak_sustained": "Peak FP32 (FMUL)",
        "sm__sass_thread_inst_executed_op_dfma_pred_on.sum.peak_sustained": "Peak FP64 (DFMA)",
        "smsp__sass_thread_inst_executed_op_fadd_pred_on.sum.per_cycle_elapsed": "Achieved FP32 FADD",
        "smsp__sass_thread_inst_executed_op_fmul_pred_on.sum.per_cycle_elapsed": "Achieved FP32 FMUL",
        "smsp__sass_thread_inst_executed_op_ffma_pred_on.sum.per_cycle_elapsed": "Achieved FP32 FFMA",
        "smsp__sass_thread_inst_executed_op_dadd_pred_on.sum.per_cycle_elapsed": "Achieved FP64 DADD",
        "smsp__sass_thread_inst_executed_op_dmul_pred_on.sum.per_cycle_elapsed": "Achieved FP64 DMUL",
        "smsp__sass_thread_inst_executed_op_dfma_pred_on.sum.per_cycle_elapsed": "Achieved FP64 DFMA",
        "sm__sass_thread_inst_executed_op_integer_pred_on.sum.peak_sustained": "Peak INT",
        "smsp__sass_thread_inst_executed_op_integer_pred_on.sum.per_cycle_elapsed": "Achieved INT",
        "smsp__cycles_elapsed.avg.per_second": "Cycles Per Second",
        "dram__bytes.sum.per_second": "DRAM Bytes Per Second",
    }


def metric_memory():
    return {
        "dram__bytes_read.sum": "Total Bytes Read",
        "dram__bytes_write.sum": "Total Bytes Write",
        "smsp__sass_inst_executed_op_memory_8b.sum": "8-bit Warp Insts",
        "smsp__sass_inst_executed_op_memory_16b.sum": "16-bit Warp Insts",
        "smsp__sass_inst_executed_op_memory_32b.sum": "32-bit Warp Insts",
        "smsp__sass_inst_executed_op_memory_64b.sum": "64-bit Warp Insts",
        "smsp__sass_inst_executed_op_memory_128b.sum": "128-bit Warp Insts",
        "l1tex__t_requests_pipe_lsu_mem_global_op_ld.sum": "L1/TEX Global Load Requests",
        "l1tex__t_requests_pipe_lsu_mem_global_op_st.sum": "L1/TEX Global Store Requests",
        "l1tex__t_requests_pipe_lsu_mem_local_op_ld.sum": "L1/TEX Local Load Requests",
        "l1tex__t_requests_pipe_lsu_mem_local_op_st.sum": "L1/TEX Local Store Requests",
        "l1tex__t_sectors_pipe_lsu_mem_global_op_ld.sum": "L1/TEX Global Load Sectors",
        "l1tex__t_sectors_pipe_lsu_mem_global_op_st.sum": "L1/TEX Global Store Sectors",
        "l1tex__t_sectors_pipe_lsu_mem_local_op_ld.sum": "L1/TEX Local Load Sectors",
        "l1tex__t_sectors_pipe_lsu_mem_local_op_st.sum": "L1/TEX Local Store Sectors",
        "lts__t_requests_srcunit_tex_op_read.sum": "L2 Read Requests",
        "lts__t_requests_srcunit_tex_op_write.sum": "L2 Write Requests",
        "lts__t_sectors_srcunit_tex_op_read.sum": "L2 Read Sectors",
        "lts__t_sectors_srcunit_tex_op_write.sum": "L2 Write Sectors",
        "dram__bytes_read.sum.pct_of_peak_sustained_elapsed": "DRAM Read Peak",
        "lts__t_sectors_srcunit_tex_op_read_lookup_hit.sum": "L2 Lookup Hit",
        "lts__t_sectors_srcunit_tex_op_read_lookup_miss.sum": "L2 Lookup Miss",
        "l1tex__t_sector_hit_rate.pct": "L1/TEX Hit Rate",
        "lts__t_sector_hit_rate.pct": "L2 Hit Rate",
    }


def metric_compute():
    return {
        "sm__inst_executed.avg.per_cycle_elapsed": "Executed IPC Elapsed",
        "sm__instruction_throughput.avg.pct_of_peak_sustained_active": "SM Busy",
        "sm__inst_executed.avg.per_cycle_active": "Executed IPC Active",
        "sm__inst_issued.avg.pct_of_peak_sustained_active": "Issue Slots Busy",
        "sm__inst_issued.avg.per_cycle_active": "Issued IPC Active",
        "sm__inst_executed_pipe_adu.avg.pct_of_peak_sustained_active": "ADU Utilization",
        "sm__pipe_alu_cycles_active.avg.pct_of_peak_sustained_active": "ALU Utilization",
        "sm__inst_executed_pipe_cbu.avg.pct_of_peak_sustained_active": "CBU Utilization",
        "sm__pipe_fma_cycles_active.avg.pct_of_peak_sustained_active": "FMA Utilization",
        "sm__inst_executed_pipe_fp16.avg.pct_of_peak_sustained_active": "FP16 Utilization",
        # "sm__inst_executed_pipe_fma_type_fp16.avg.pct_of_peak_sustained_active": "FMA (FP16) Utilization",
        "sm__inst_executed_pipe_fp64.avg.pct_of_peak_sustained_active": "FP64 Utilization",
        "sm__inst_executed_pipe_lsu.avg.pct_of_peak_sustained_active": "LSU Utilization",
        "sm__inst_executed_pipe_tensor_op_dmma.avg.pct_of_peak_sustained_active": "Tensor (DP) Utilization",
        "sm__inst_executed_pipe_tensor_op_hmma.avg.pct_of_peak_sustained_active": "Tensor (FP) Utilization",
        "sm__inst_executed_pipe_tensor_op_imma.avg.pct_of_peak_sustained_active": "Tensor (INT) Utilization",
        "sm__inst_executed_pipe_tex.avg.pct_of_peak_sustained_active": "Tex Utilization",
        "sm__inst_executed_pipe_uniform.avg.pct_of_peak_sustained_active": "Uniform Utilization",
        "sm__inst_executed_pipe_xu.avg.pct_of_peak_sustained_active": "XU Utilization",
    }


def metric_occupancy():
    return {
        "sm__maximum_warps_per_active_cycle_pct": "Theoretical Occupancy",
        "sm__maximum_warps_avg_per_active_cycle": "Theoretical Active Warps per SM",
        "sm__warps_active.avg.pct_of_peak_sustained_active": "Achieved Occupancy",
        "sm__warps_active.avg.per_cycle_active": "Achieved Active Warps per SM",
        "launch__occupancy_limit_registers": "Block Limit Register",
        "launch__occupancy_limit_shared_mem": "Block Limit Shared Mem",
        "launch__occupancy_limit_warps": "Block Limit Warp",
        "launch__occupancy_limit_blocks": "Block Limit SM",
    }


def metric_launch():
    return {
        "launch__thread_count": "Threads",
        "launch__grid_size": "Grid Size",
        "launch__block_size": "Block Size",
        "launch__registers_per_thread": "Register Per Thread",
        "launch__waves_per_multiprocessor": "Waves Per SM",
    }


def metric_warp():
    return {
        "smsp__average_warp_latency_per_inst_issued.ratio": "Warp Cycles Per Issued Instruction",
        "smsp__average_warps_active_per_inst_executed.ratio": "Warp Cycles Per Executed Instruction",
        "smsp__thread_inst_executed_per_inst_executed.ratio": "Avg. Active Threads Per Warp",
        "smsp__thread_inst_executed_pred_on_per_inst_executed.ratio": "Avg. Not Predicated Off Threads Per Warp",
        "smsp__warps_active.avg.peak_sustained": "GPU Max Warps Per Scheduler",
        "smsp__maximum_warps_avg_per_active_cycle": "Theoretical Warps Per Scheduler",
        "smsp__warps_active.avg.per_cycle_active": "Active Warps Per Scheduler",
        "smsp__warps_eligible.avg.per_cycle_active": "Eligible Warps Per Scheduler",
        "smsp__issue_active.avg.per_cycle_active": "Issued Warps Per Scheduler",
    }


def metric_detail_warp():
    return {
        "smsp__average_warps_issue_stalled_drain_per_issue_active.ratio": "Stall Drain",
        "smsp__average_warps_issue_stalled_imc_miss_per_issue_active.ratio": "Stall IMC Miss",
        "smsp__average_warps_issue_stalled_barrier_per_issue_active.ratio": "Stall Barrier",
        "smsp__average_warps_issue_stalled_branch_resolving_per_issue_active.ratio": "Stall Branch Resolving",
        "smsp__average_warps_issue_stalled_membar_per_issue_active.ratio": "Stall Membar",
        "smsp__average_warps_issue_stalled_short_scoreboard_per_issue_active.ratio": "Stall Short Scoreboard",
        "smsp__average_warps_issue_stalled_sleeping_per_issue_active.ratio": "Stall Sleep",
        "smsp__average_warps_issue_stalled_wait_per_issue_active.ratio": "Stall Wait",
        "smsp__average_warps_issue_stalled_no_instruction_per_issue_active.ratio": "Stall No Instruction",
        "smsp__average_warps_issue_stalled_math_pipe_throttle_per_issue_active.ratio": "Stall Math Pipe Throttle",
        "smsp__average_warps_issue_stalled_tex_throttle_per_issue_active.ratio": "Stall Tex Throttle",
        "smsp__average_warps_issue_stalled_lg_throttle_per_issue_active.ratio": "Stall LG Throttle",
        "smsp__average_warps_issue_stalled_dispatch_stall_per_issue_active.ratio": "Stall Dispatch Stall",
        "smsp__average_warps_issue_stalled_misc_per_issue_active.ratio": "Stall Misc",
        "smsp__average_warps_issue_stalled_not_selected_per_issue_active.ratio": "Stall Not Selected",
        "smsp__average_warps_issue_stalled_selected_per_issue_active.ratio": "Stall Selected",
        "smsp__average_warps_issue_stalled_long_scoreboard_per_issue_active.ratio": "Stall Long Scoreboard",
        "smsp__average_warps_issue_stalled_mio_throttle_per_issue_active.ratio": "Stall MIO Throttle",
    }


def metric_inst():
    return {
        "smsp__inst_executed.sum": "Executed Insts",
        "smsp__inst_executed.avg": "Avg. Executed Insts Per Scheduler",
        "smsp__inst_issued.sum": "Issued Insts",
        "smsp__inst_issued.avg": "Avg. Issued Insts Per Scheduler",
        "smsp__thread_inst_executed_per_inst_executed.ratio": "Avg. Threads Per Inst",
        "smsp__thread_inst_executed_pred_on_per_inst_executed.ratio": "Avg. Active Predicated-On Threads Per Inst",
    }


def all_metric():
    return {
        **metric_sol(),
        **metric_roofline(),
        **metric_memory(),
        **metric_compute(),
        **metric_occupancy(),
        **metric_launch(),
        **metric_warp(),
        **metric_detail_warp(),
        **metric_inst(),
    }
