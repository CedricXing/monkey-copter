import numpy as np


def trace_of(raw_file, return_dict=True):
    if return_dict:
        td = {}
        with open(raw_file) as f:
            for line in f:
                if ":" not in line:
                    continue
                td[line.split(":")[0]] = int(line.split(":")[1])
        return td
    else:
        ts = set()
        with open(raw_file) as f:
            for line in f:
                if ":" not in line:
                    continue
                ts.add(line.split(":")[0])
        return ts


def state_of(npy_file, ignore_last=0):
    s = np.load(npy_file)
    if ignore_last == 0:
        return s
    else:
        return s[:-ignore_last, :]

