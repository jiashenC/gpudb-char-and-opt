import os
import subprocess as subp

from collections import defaultdict

from utility.profiler_logger import LOGGER


def convert_unit(input_str):
    value, unit = input_str.split(" ")

    value = float(value)

    # cycles per second conversion
    if unit == "cycle/nsecond":
        value = value * 1000000000.0
        unit = "cycle/second"
    elif unit == "cycle/usecond":
        value = value * 1000000.0
        unit = "cycle/second"
    elif unit == "cycle/msecond":
        value = value * 1000.0
        unit = "cycle/second"
    elif unit == "cycle/second":
        value = value
        unit = "cycle/second"

    # bytes per second conversion
    elif unit == "byte/second":
        value = value
        unit = "byte/second"
    elif unit == "Kbyte/second":
        value = value * 1024.0
        unit = "byte/second"
    elif unit == "Mbyte/second":
        value = value * (1024.0**2)
        unit = "byte/second"
    elif unit == "Gbyte/second":
        value = value * (1024.0**3)
        unit = "byte/second"
    elif unit == "Tbyte/second":
        value = value * (1024.0**4)
        unit = "byte/second"

    # bytes conversion
    elif unit == "byte":
        value = value
        unit = "byte"
    elif unit == "Kbyte":
        value = value * 1024.0
        unit = "byte"
    elif unit == "Mbyte":
        value = value * (1024.0**2)
        unit = "byte"
    elif unit == "Gbyte":
        value = value * (1024.0**3)
        unit = "byte"
    elif unit == "Tbyte":
        value = value * (1024.0**4)
        unit = "byte"

    # seconds conversion
    elif unit == "usecond":
        value = value / 1000000.0
        unit = "second"
    elif unit == "msecond":
        value = value / 1000.0
        unit = "second"
    elif unit == "second":
        value = value / 1000.0
        unit = "second"

    # units no need for conversion
    elif unit == "%":
        value = value
    elif unit == "block":
        value = value
    elif unit == "cycle":
        value = value
    elif unit == "request":
        value = value
    elif unit == "sector":
        value = value
    elif unit == "inst":
        value = value
    elif unit == "warp":
        value = value
    elif unit == "cycle":
        value = value
    elif unit == "register/thread":
        value = value
    elif unit == "thread":
        value = value
    elif unit == "inst/cycle":
        value = value
    elif unit == "":
        value = value

    # handle exception
    else:
        raise Exception(f"{unit} is not supported.")

    return value, unit


class NcuParser:
    def __init__(self, path="."):
        LOGGER.debug("Ncu parser starts parsing")

        p = subp.Popen(
            [
                "ncu",
                "--import",
                os.path.join(path, "gpudb-perf.ncu-rep"),
                "--csv",
            ],
            stdin=subp.PIPE,
            stdout=subp.PIPE,
            stderr=subp.STDOUT,
        )
        out, _ = p.communicate()
        out = out.decode("utf-8")

        self._ordered_kernel_list = []

        self._fn_to_idx = dict()
        self._metric_dict = defaultdict(lambda: defaultdict(lambda: float))

        parse_metric = False
        metric_cnt = 0

        for line in out.split("\n"):
            line = line.strip(",").strip('"').split('","')

            if not parse_metric:
                # skip first few lines (warnings)
                if line[0] != "ID":
                    LOGGER.info(line)
                    continue

                # build field name to idx
                for idx, fn in enumerate(line):
                    self._fn_to_idx[fn] = idx

                parse_metric = True
                metric_cnt = len(line)
            else:
                if len(line) != metric_cnt:
                    continue
                kernel_name = line[self._fn_to_idx["Kernel Name"]]
                metric_name = line[self._fn_to_idx["Metric Name"]]
                raw_value = line[self._fn_to_idx["Metric Value"]].replace(
                    ",", ""
                )
                metric_value = 0.0 if raw_value == "n/a" else float(raw_value)
                metric_unit = line[self._fn_to_idx["Metric Unit"]]

                # handle for kernels being called multiple times
                idx = 0
                while True:
                    tmp_kernel_name = f"{kernel_name}_{idx}"
                    if metric_name not in self._metric_dict[tmp_kernel_name]:
                        kernel_name = tmp_kernel_name
                        break
                    idx += 1

                if (
                    len(self._ordered_kernel_list) == 0
                    or kernel_name != self._ordered_kernel_list[-1]
                ):
                    self._ordered_kernel_list.append(kernel_name)

                self._metric_dict[kernel_name][
                    metric_name
                ] = f"{metric_value} {metric_unit}"

    def gen_res(self, kn, fn_list):
        res = []

        for fn in fn_list:
            if fn not in self._metric_dict[kn]:
                continue
            value_str = self._metric_dict[kn][fn]
            value, unit = convert_unit(value_str)
            res.append((fn, "{} {}".format(value, unit)))

        return res

    def get_kernel_list(self):
        return self._ordered_kernel_list

    def get_field_list(self):
        kn = list(self._metric_dict.keys())[0]
        return list(self._metric_dict[kn].keys())

    def get_unit(self, kn, fn):
        value_str = self._metric_dict[kn][fn]
        _, unit = convert_unit(value_str)
        return unit

    def get_value(self, kn, fn):
        value_str = self._metric_dict[kn][fn]
        value, _ = convert_unit(value_str)
        return value
