from testclassification2 import *
from reportgen2 import *


def evaluate_report_top(report, top_max_number=15):
    scores = []
    ground_truth = set()
    ground_truth.add(4)
    ground_truth.add(5)
    ground_truth.add(6)
    hit_count = 0
    miss_count = len(ground_truth)
    wrong_count = 0
    for k, v in report[0:top_max_number]:
        if k in ground_truth:
            hit_count += 1
            miss_count -= 1
        else:
            wrong_count += 1
        precesion = float(hit_count) / (hit_count + wrong_count)
        recall = float(hit_count) / (hit_count + miss_count)
        scores.append((precesion, recall))
    return scores


DATA_DIR = '/Users/enyanhuang/Developer/PycharmProjects/ArdupilotDebug/data/small_exp3/'
valid_sims = range(1000)
pass_cb, fail_cb = constrain_based(DATA_DIR+'AB/', valid_sims)
pass_dr, fail_dr = divergence_ratio_based(DATA_DIR+'PA/', valid_sims)

report_cb_ta = tarantula(DATA_DIR+'AB/', pass_cb, fail_cb)
report_dr_ta = tarantula(DATA_DIR+'PA/', pass_dr, fail_dr)
report_cb_cr = crosstab(DATA_DIR+'AB/', pass_cb, fail_cb)
report_dr_cr = crosstab(DATA_DIR+'PA/', pass_dr, fail_dr)
report_cb_bpnn = bpnn(DATA_DIR+'AB/', pass_cb, fail_cb)
report_dr_bpnn = bpnn(DATA_DIR+'PA/', pass_dr, fail_dr)

scores_cb_ta = evaluate_report_top(report_cb_ta)
scores_dr_ta = evaluate_report_top(report_dr_ta)
scores_cb_cr = evaluate_report_top(report_cb_cr)
scores_dr_cr = evaluate_report_top(report_dr_cr)
scores_cb_bpnn = evaluate_report_top(report_cb_bpnn)
scores_dr_bpnn = evaluate_report_top(report_dr_bpnn)
print "CB TA:"
print scores_cb_ta
print "DR TA:"
print scores_dr_ta
print "CB CR:"
print scores_cb_cr
print "DR CR:"
print scores_dr_ta
print "CB BPNN:"
print scores_cb_bpnn
print "DR BPNN"
print scores_dr_bpnn
