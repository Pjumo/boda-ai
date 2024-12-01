import os
import glob
import json
import csv
import numpy as np
import time
import datetime


class timer():
    """Class for measuring executed time."""

    def __init__(self):
        self.acc = 0
        self.tic()

    def tic(self):
        self.t0 = time.time()

    def toc(self, restart=False):
        diff = time.time() - self.t0
        if restart:
            self.t0 = time.time()
        return diff

    def hold(self):
        self.acc += self.toc()

    def release(self):
        ret = self.acc
        self.acc = 0
        return ret

    def reset(self):
        self.acc = 0


def myconverter(obj):
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, datetime.datetime):
        return obj.__str__()
    else:
        return obj


def write_csv(results_list, file_name):
    result_dir = os.path.dirname(file_name)
    if result_dir != "":
        os.makedirs(result_dir, exist_ok=True)
    headers = ["Model Name", "Batch Size", "Calibration Table", "mAP@[0.5:0.95]", "mAP@[0.5]", "Execution Time(s)", "Latency(ms)", "Throughput(fps)", "Note"]
    with open(file_name, "a") as f:
        file_is_empty = os.stat(file_name).st_size == 0
        w = csv.writer(f, delimiter=",")
        if file_is_empty:
            w.writerow(headers)
        w.writerow(results_list)
        f.close()


def make_table(calib_table: str, calibrations: dict, mode: str):
    calib_table_dir = os.path.dirname(calib_table)
    if calib_table_dir != '':
        os.makedirs(calib_table_dir, exist_ok=True)
    if 'x330' in mode:
        with open(calib_table, 'w') as json_file:
            json.dump(calibrations, json_file, indent=4)
    else:
        quant_file = open(calib_table, "w")
        for name, thrshld in calibrations.items():
            fmt = "%s\t%f\n" % (name, thrshld)
            quant_file.write(fmt)
        quant_file.close()
    print("Save Calibration table file to >>> {}".format(calib_table))

def sapeon_device_id(device="x330"):
    device_cnt = 0
    if device == "x330":
        device_path = "/dev/snx3-*"
    else:
        device_path = "/dev/aix*"
    for name in glob.glob(device_path):
        device_id = name[-1]
        device_cnt = device_cnt +1
    assert device_cnt > 0, "Please check SAPEON devices, there is no SAPEON device"
    return device_id

def profiling(device="x330"):
    os.environ["SAPEONRT_ENABLE_FS"] = "9999"
    if device == "x330":
        os.environ["AIXV_FS_DRY_MODE"] = "1"        
    else:
        os.environ["AIXH_FS_DRY_MODE"] = "1"
