from dataclean import *
from testclassification import *
from reportgen import *
import pickle
import sys
from getopt import getopt, GetoptError
import os


def evaluate_report(report, top_max_number=15):
    scores = []
    ground_truth = set()
    with open("%sbug_location.txt" % exp_data_dir) as ground_truth_file:
        for line in ground_truth_file:
            ground_truth.add("/home/else/Documents/ardupilot/"+line.strip("\n"))
    hit_count = 0
    miss_count = len(ground_truth)
    wrong_count = 0
    i = 0;
    print(report[:15])
    while (top_max_number!=0):
        k, v = report[i];
        if (not k.startswith("/home/else/Documents/ardupilot/libraries")):
            i += 1
            continue
        if k in ground_truth:
            print("Here", i, top_max_number, k)
            hit_count += 1
            miss_count -= 1
        else:
            wrong_count += 1
        i += 1
        top_max_number -= 1
    # for k, v in report[0:top_max_number]:
    #     print(report[0]);
    #     if k in ground_truth:
    #         hit_count += 1
    #         miss_count -= 1
    #     else:
    #         wrong_count += 1
    ####################???????????????????????????#############
    precesion = float(hit_count) / (hit_count + wrong_count)
    recall = float(hit_count) / (hit_count + miss_count)
    scores.append((precesion, recall))
    
    return scores


def print_usage(filename):
    print("Usage: %s --data_dirs=datadir1,datadir2[,...] --out_dir=outdir"
          % filename)


if __name__ == '__main__':
    try:
        opts, args = getopt(sys.argv[1:], 'h', ['data_dirs=', 'out_dir=', 'help'])
    except GetoptError as e:
        print_usage(sys.argv[0])
        sys.exit(2)

    data_dirs = None
    out_dir = None
    for opt, arg in opts:
        if opt in ['-h', '--help']:
            print_usage(sys.argv[0])
            sys.exit(0)
        elif opt == '--data_dirs':
            data_dirs = arg.split(',')
            for d in data_dirs:
                d = d.strip()
                if not d.endswith('/'):
                    d += '/'
        elif opt == '--out_dir':
            out_dir = arg
    if data_dirs is None or out_dir is None:
        print_usage(sys.argv[0])
        sys.exit(2)

    for data_dir in data_dirs:
        for partition_i in range(2):
            for tu in [15, 30, 45, 60, 75, 90, 105, 120]:
                su = int(float(tu) / 12 * 40)
                scores_ta_cb = []
                scores_cr_cb = []
                scores_bpnn_cb = []
                scores_ta_dr = []
                scores_cr_dr = []
                scores_bpnn_dr = []

                reports = {}
                labeling = {}
                for labeling_method in ['PA']:  #for labeling_method in ['AB', 'PA']:
                    exp_dir = "%s%s/" % (data_dir, labeling_method)

                    for exp_i in os.listdir(exp_dir):
                        if exp_i.startswith('.'):
                            continue
                        exp_data_dir = '%s%s/' % (exp_dir, exp_i)
                        print("Analysing data in %s" % exp_data_dir)
                        if partition_i == 0:
                            valid_simulations = filter_by_raw(exp_data_dir, index_from=0, index_to=1000)
                        else:
                            valid_simulations = filter_by_raw(exp_data_dir, index_from=1000, index_to=2000)

                        if labeling_method == 'AB':
                            pass_cases, fail_cases = constrain_based(exp_data_dir, valid_simulations, states_until=su)
                        else:
                            pass_cases, fail_cases = divergence_ratio_based(exp_data_dir, valid_simulations,
                                                                            states_until=su)
                        labeling['%s_%s' % (labeling_method, exp_i)] = {'pass': pass_cases, 'fail': fail_cases}
                        report_tarantula = tarantula(exp_data_dir, pass_cases, fail_cases)
                        report_crosstab = crosstab(exp_data_dir, pass_cases, fail_cases)
                        report_bpnn = bpnn(exp_data_dir, pass_cases, fail_cases)
                        reports['%s_TA_%s' % (labeling_method, exp_i)] = report_tarantula
                        reports['%s_CR_%s' % (labeling_method, exp_i)] = report_crosstab
                        reports['%s_NN_%s' % (labeling_method, exp_i)] = report_bpnn
                        
                        if labeling_method == 'AB':
                            scores_ta_cb.append(evaluate_report(report_tarantula))
                            scores_cr_cb.append(evaluate_report(report_crosstab))
                            scores_bpnn_cb.append(evaluate_report(report_bpnn))
                        else:
                            scores_ta_dr.append(evaluate_report(report_tarantula))
                            scores_cr_dr.append(evaluate_report(report_crosstab))
                            scores_bpnn_dr.append(evaluate_report(report_bpnn))

                exp_name = os.path.basename(os.path.normpath(data_dir))
                with open("%sresult_%s_p%d_%d.pckl" % (out_dir, exp_name, partition_i+1, tu), 'wb') as fp:
                    pickle.dump({"Tarantula, constrain-based": scores_ta_cb,
                                 "Tarantula, divergence ratio": scores_ta_dr,
                                 "Crosstab, constrain-based": scores_cr_cb,
                                 "Crosstab, divergence ratio": scores_cr_dr,
                                 "BPNN, constrain-based": scores_bpnn_cb,
                                 "BPNN, divergence ratio": scores_bpnn_dr}, fp)
                    print("Tarantula, constrain-based", scores_ta_cb,
                                 "Tarantula, divergence ratio", scores_ta_dr,
                                 "Crosstab, constrain-based", scores_cr_cb,
                                 "Crosstab, divergence ratio", scores_cr_dr,
                                 "BPNN, constrain-based", scores_bpnn_cb,
                                 "BPNN, divergence ratio", scores_bpnn_dr)
                with open("%slabel_%s_p%d_%d.pckl" % (out_dir, exp_name, partition_i+1, tu), 'wb') as fp:
                    pickle.dump(labeling, fp)
                with open("%sreport_%s_p%d_%d.pckl" % (out_dir, exp_name, partition_i+1, tu), 'wb') as fp:
                    pickle.dump(reports, fp)