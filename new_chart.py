import pickle
import numpy as np
from scipy.stats import wilcoxon
import matplotlib.pyplot as plt
import matplotlib

DATA_DIR = '/Users/enyanhuang/Developer/ArdupilotDebugData/big_exp_result/1bug_95/'
DATA_DIR2 = '/Users/enyanhuang/Developer/ArdupilotDebugData/big_exp_result/5bug_95/'

ts = [15, 30, 45, 60, 75, 90, 105, 120]
ttds_time = {}
precesions_time = {}
recalls_time = {}

for t in ts:
    print "T=%d" % t
    results = []
    labels = []
    with open(DATA_DIR+'result_big_exp_hover_1bug_p1_%d.pckl' % t) as f:
        results.append(pickle.load(f))
    with open(DATA_DIR+'result_big_exp_hover_1bug_p2_%d.pckl' % t) as f:
        results.append(pickle.load(f))
    with open(DATA_DIR+'result_big_exp_no_hover_1bug_p1_%d.pckl' % t) as f:
        results.append(pickle.load(f))
    with open(DATA_DIR+'result_big_exp_no_hover_1bug_p2_%d.pckl' % t) as f:
        results.append(pickle.load(f))

    with open(DATA_DIR2+'result_big_exp_hover_5bug_p1_%d.pckl' % t) as f:
        results.append(pickle.load(f))
    with open(DATA_DIR2+'result_big_exp_hover_5bug_p2_%d.pckl' % t) as f:
        results.append(pickle.load(f))
    with open(DATA_DIR2+'result_big_exp_no_hover_5bug_p1_%d.pckl' % t) as f:
        results.append(pickle.load(f))
    with open(DATA_DIR2+'result_big_exp_no_hover_5bug_p2_%d.pckl' % t) as f:
        results.append(pickle.load(f))

    precesions = {}
    recalls = {}
    ttds = {}

    # Calculate precesion
    for r_i, r in enumerate(results):
        if r_i < 4:
            v_start = 0
            v_end = 12
        else:
            v_start = 0
            v_end = 35
        for k, v in r.items():
            for version_i, top_list in enumerate(v[v_start:v_end]):
                discoverged = False
                for top_i, pr_pair in enumerate(top_list[0:10]):
                    if k not in ttds.keys():
                        ttds[k] = []
                    if pr_pair[0] != 0 and not discoverged:
                        ttds[k].append(top_i+1)
                        discoverged = True
                    if top_i == 9:
                        if k not in precesions.keys():
                            precesions[k] = []
                            recalls[k] = []
                        precesions[k].append(pr_pair[0])
                        recalls[k].append(pr_pair[1])
                if not discoverged:
                    ttds[k].append(10)
    ttds_time[t] = ttds
    precesions_time[t] = precesions
    recalls_time[t] = recalls
    # labels_time[t] = labels

# Calculate p value and ES
precesions = precesions_time[120]
recalls = recalls_time[120]
ttds = ttds_time[120]
for method in ['Tarantula, divergence ratio', 'Tarantula, constrain-based',
               'Crosstab, divergence ratio', 'Crosstab, constrain-based',
               'BPNN, divergence ratio', 'BPNN, constrain-based']:
    if method.endswith('divergence ratio'):
        print method
        ab_precesion = np.array(precesions[method.replace('divergence ratio', 'constrain-based')])
        pa_precesion = np.array(precesions[method])
        ab_recall = np.array(recalls[method.replace('divergence ratio', 'constrain-based')])
        pa_recall = np.array(recalls[method])
        ab_ttd = np.array(ttds[method.replace('divergence ratio', 'constrain-based')])
        pa_ttd = np.array(ttds[method])
        _, pvalue_precesion = wilcoxon(ab_precesion, pa_precesion)
        _, pvalue_recall = wilcoxon(ab_recall, pa_recall)
        _, pvalue_ttd = wilcoxon(ab_ttd, pa_ttd)
        std_ab_precesion = np.std(ab_precesion)
        std_pa_precesion = np.std(pa_precesion)
        std_ab_recall = np.std(ab_recall)
        std_pa_recall = np.std(pa_recall)
        std_ab_ttd = np.std(ab_ttd)
        std_pa_ttd = np.std(pa_ttd)
        s_precesion = np.sqrt(((len(ab_precesion) - 1) * (std_ab_precesion ** 2) + (len(pa_precesion) - 1) * (std_pa_precesion ** 2)) / (len(ab_precesion) + len(pa_precesion) - 2))
        s_recall = np.sqrt(
            ((len(ab_recall) - 1) * (std_ab_recall ** 2) + (len(pa_recall) - 1) * (std_pa_recall ** 2)) / (
            len(ab_recall) + len(pa_recall) - 2))
        s_ttd = np.sqrt(((len(ab_ttd) - 1) * (std_ab_ttd ** 2) + (len(pa_ttd) - 1) * (std_pa_ttd ** 2)) / (len(ab_ttd) + len(pa_ttd) - 2))
        d_precesion = (np.mean(pa_precesion) - np.mean(ab_precesion)) / s_precesion
        d_recall = (np.mean(pa_recall) - np.mean(ab_recall)) / s_recall
        d_ttd = (np.mean(pa_ttd) - np.mean(ab_ttd)) / s_ttd
        # print d_precesion
        # print d_recall
        # print d_ttd
        # print (pvalue_precesion / 2)
        # print (pvalue_recall / 2)
        # print (pvalue_ttd / 2)
        # print "Improvement: "
        # print np.mean(pa_precesion) / np.mean(ab_precesion)
        # print np.mean(pa_recall) / np.mean(ab_recall)
        # print np.mean(ab_ttd) / np.mean(pa_ttd)

data_x = [15, 30, 45, 60, 75, 90, 105, 120]
data_precesion = {}
data_ttd = {}
data_recall = {}

for x in data_x:
    precesions = precesions_time[x]
    for method, ps in precesions.items():
        if method not in data_precesion.keys():
            data_precesion[method] = []
        data_precesion[method].append(np.mean(ps))

for method in ['Tarantula, divergence ratio', 'Tarantula, constrain-based',
               'Crosstab, divergence ratio', 'Crosstab, constrain-based',
               'BPNN, divergence ratio', 'BPNN, constrain-based']:
    if method.endswith('divergence ratio'):
        print method
        ab = np.array(data_precesion[method.replace('divergence ratio', 'constrain-based')])
        pa = np.array(data_precesion[method])
        _, pvalue = wilcoxon(ab, pa)
        std_ab = np.std(ab)
        std_pa = np.std(pa)
        s = np.sqrt(
            ((len(ab) - 1) * (std_ab ** 2) + (len(pa) - 1) * (std_pa ** 2)) / (
            len(ab) + len(pa) - 2))
        d = (np.mean(pa) - np.mean(ab)) / s

        print (pvalue / 2)
        print d

for x in data_x:
    recalls = recalls_time[x]
    for method, ps in recalls.items():
        if method not in data_recall.keys():
            data_recall[method] = []
        data_recall[method].append(np.mean(ps))

for method in ['Tarantula, divergence ratio', 'Tarantula, constrain-based',
               'Crosstab, divergence ratio', 'Crosstab, constrain-based',
               'BPNN, divergence ratio', 'BPNN, constrain-based']:
    if method.endswith('divergence ratio'):
        print method
        ab = np.array(data_recall[method.replace('divergence ratio', 'constrain-based')])
        pa = np.array(data_recall[method])
        _, pvalue = wilcoxon(ab, pa)
        std_ab = np.std(ab)
        std_pa = np.std(pa)
        s = np.sqrt(
            ((len(ab) - 1) * (std_ab ** 2) + (len(pa) - 1) * (std_pa ** 2)) / (
            len(ab) + len(pa) - 2))
        d = (np.mean(pa) - np.mean(ab)) / s

        print (pvalue / 2)
        print d

for x in data_x:
    ttds = ttds_time[x]
    for method, ps in ttds.items():
        if method not in data_ttd.keys():
            data_ttd[method] = []
        data_ttd[method].append(np.mean(ps))

for method in ['Tarantula, divergence ratio', 'Tarantula, constrain-based',
               'Crosstab, divergence ratio', 'Crosstab, constrain-based',
               'BPNN, divergence ratio', 'BPNN, constrain-based']:
    if method.endswith('divergence ratio'):
        print method
        ab = np.array(data_ttd[method.replace('divergence ratio', 'constrain-based')])
        pa = np.array(data_ttd[method])
        _, pvalue = wilcoxon(ab, pa)
        std_ab = np.std(ab)
        std_pa = np.std(pa)
        s = np.sqrt(
            ((len(ab) - 1) * (std_ab ** 2) + (len(pa) - 1) * (std_pa ** 2)) / (
            len(ab) + len(pa) - 2))
        d = (np.mean(pa) - np.mean(ab)) / s

        print (pvalue / 2)
        print d

format_dict = {'Tarantula, divergence ratio': 'k-o',
               'Tarantula, constrain-based': 'k-^',
               'Crosstab, divergence ratio': 'k--o',
               'Crosstab, constrain-based': 'k--^',
               'BPNN, divergence ratio': 'k:o',
               'BPNN, constrain-based': 'k:^'}
label_dict = {'Tarantula, divergence ratio': 'A+TA',
              'Tarantula, constrain-based': 'B+TA',
              'Crosstab, divergence ratio': 'A+CR',
              'Crosstab, constrain-based': 'B+CR',
              'BPNN, divergence ratio': 'A+NN',
              'BPNN, constrain-based': 'B+NN'}

matplotlib.rc('xtick', labelsize=30)
matplotlib.rc('ytick', labelsize=30)
matplotlib.rc('legend', fontsize=30)

plt.figure(1)
ax = plt.subplot(111)
plt.xticks([15, 45, 75, 105])
plt.xlim([12, 123])
plt.ylim([0, 0.2])
plt.yticks([0, 0.04, 0.08, 0.12, 0.16, 0.2])
plots = []
for k in ['Tarantula, divergence ratio', 'Tarantula, constrain-based',
               'Crosstab, divergence ratio', 'Crosstab, constrain-based']:
    v = data_precesion[k]
    plot, = ax.plot(data_x, v, format_dict[k], label=label_dict[k], lw=3, markersize=15)
    plots.append(plot)
box = ax.get_position()
ax.set_position([box.x0, box.y0, box.width, box.height])

plt.figure(2)
ax = plt.subplot(111)
plt.xticks([15, 45, 75, 105])
plt.xlim([12, 123])
plt.ylim([0, 0.5])
plt.yticks([0, 0.1, 0.2, 0.3, 0.4, 0.5])
plots = []
for k in ['Tarantula, divergence ratio', 'Tarantula, constrain-based',
               'Crosstab, divergence ratio', 'Crosstab, constrain-based']:
    v = data_recall[k]
    plot, = ax.plot(data_x, v, format_dict[k], label=label_dict[k], lw=3, markersize=15)
    plots.append(plot)
box = ax.get_position()
ax.set_position([box.x0, box.y0, box.width, box.height])

plt.figure(3)
ax = plt.subplot(111)
plt.xlim([12, 123])
plt.ylim([0, 10])
plt.xticks([15, 45, 75, 105])
plots = []
for k in ['Tarantula, divergence ratio', 'Tarantula, constrain-based',
               'Crosstab, divergence ratio', 'Crosstab, constrain-based']:
    v = data_ttd[k]
    plot, = ax.plot(data_x, v, format_dict[k], label=label_dict[k], lw=3, markersize=15)
    plots.append(plot)
box = ax.get_position()
ax.set_position([box.x0, box.y0, box.width, box.height])
ax.legend(loc=3, ncol=6, mode="expand", borderaxespad=0)
plt.tight_layout()

matplotlib.rc('xtick', labelsize=30)
matplotlib.rc('ytick', labelsize=30)
matplotlib.rc('legend', fontsize=30)

plt.figure(4)
plt.ylim([-0.01, 0.51])
plt.xticks(rotation=30)
plt.yticks([0, 0.1, 0.2, 0.3, 0.4, 0.5])
precesions = precesions_time[120]
data_box = []
for method in ['Tarantula, divergence ratio', 'Tarantula, constrain-based',
               'Crosstab, divergence ratio', 'Crosstab, constrain-based',
               'BPNN, divergence ratio', 'BPNN, constrain-based']:
    data_box.append(precesions[method])
bp = plt.boxplot(data_box, labels=['A+TA', 'B+TA', 'A+CR', 'B+CR', 'A+NN', 'B+NN'],
                 positions=[2, 3, 5, 6, 8, 9],
                 patch_artist=True, medianprops={'linewidth': 4, 'color': 'black'})
colors = ['lightgray', 'white', 'lightgray', 'white', 'lightgray', 'white']
for patch, color in zip(bp['boxes'], colors):
    patch.set_facecolor(color)
plt.tight_layout()

plt.figure(5)
plt.ylim([-0.2, 11.2])
plt.xticks(rotation=30)
plt.yticks([0, 2, 4, 6, 8, 10])
ttds = ttds_time[120]
data_box = []
for method in ['Tarantula, divergence ratio', 'Tarantula, constrain-based',
               'Crosstab, divergence ratio', 'Crosstab, constrain-based',
               'BPNN, divergence ratio', 'BPNN, constrain-based']:
    data_box.append(ttds[method])
bp = plt.boxplot(data_box, labels=['A+TA', 'B+TA', 'A+CR', 'B+CR', 'A+NN', 'B+NN'],
                 positions=[2, 3, 5, 6, 8, 9],
                 patch_artist=True, medianprops={'linewidth': 4, 'color': 'black'})
colors = ['lightgray', 'white', 'lightgray', 'white', 'lightgray', 'white']
for patch, color in zip(bp['boxes'], colors):
    patch.set_facecolor(color)
plt.tight_layout()

plt.figure(6)
plt.ylim([-0.02, 1.02])
plt.yticks([0, 0.2, 0.4, 0.6, 0.8, 1])
plt.xticks(rotation=30)
recalls = recalls_time[120]
data_box = []
for method in ['Tarantula, divergence ratio', 'Tarantula, constrain-based',
               'Crosstab, divergence ratio', 'Crosstab, constrain-based',
               'BPNN, divergence ratio', 'BPNN, constrain-based']:
    data_box.append(recalls[method])
bp = plt.boxplot(data_box, labels=['A+TA', 'B+TA', 'A+CR', 'B+CR', 'A+NN', 'B+NN'],
                 positions=[2, 3, 5, 6, 8, 9],
                 patch_artist=True, medianprops={'linewidth': 4, 'color': 'black'})
colors = ['lightgray', 'white', 'lightgray', 'white', 'lightgray', 'white']
for patch, color in zip(bp['boxes'], colors):
    patch.set_facecolor(color)
plt.tight_layout()

plt.show()
