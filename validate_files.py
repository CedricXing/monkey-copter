import pickle
import numpy as np


for exp_i in range(20, 30):
    print("Exp %d" % exp_i)
    DATA_DIR = '/Users/enyanhuang/Developer/PycharmProjects/ArdupilotDebug/data/big_exp2/%d/' % exp_i
    for sim_i in range(2000):
        try:
            with open(DATA_DIR+'profiles_%d.pckl' % sim_i) as profile_file:
                pickle.load(profile_file)
        except:
            print("Cannot open profile file %d" % sim_i)
        for core_i in range(4):
            try:
                np.load(DATA_DIR+'states_%d_%d.npy' % (sim_i, core_i))
            except:
                print("Cannot open states file %d_%d" % (sim_i, core_i))

print('Validation complete.')