import matplotlib.pyplot as plt
import numpy as np

DATA_DIR = '/Users/enyanhuang/Developer/PycharmProjects/ArdupilotDebug/data/small_exp2/0/'
states1 = np.load(DATA_DIR+'states_13_2.npy')
states2 = np.load(DATA_DIR+'states_13_3.npy')

plt.plot(states1)
plt.plot(states2)

plt.show()
