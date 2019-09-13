import os
from injector import *
import time
from runsimulation import *
from runsimul_extend import run_sim_extend
import random
from ConfigParser import ConfigParser

dir_compile = '/home/cedric/Desktop/arduPilot/'
dir_monkey = 'home/cedric/Desktop/arduPilot/monkey-copter/'

def parserConfig():
    cfg = ConfigParser()
    cfg.read('/home/cedric/record/config.ini')
    real_life,mutiple_bugs = cfg.get('param','real_life'),cfg.get('param','mutiple_bugs')
    mutiple_bugs_start = cfg.get('param','mutiple_bugs_start')
    return real_life,mutiple_bugs

def recoverAllFiles():
    for bug in bug_group:
        os.system('cp /home/cedric/Desktop/ArduPilot_Back/' + bug['file'] + ' /home/cedric/Desktop/arduPilot/'+bug['file'])
    for bug in real_life_bug_group:
        os.system('cp /home/cedric/Desktop/ArduPilot_Back/' + bug['file'] + ' /home/cedric/Desktop/arduPilot/'+bug['file'])
        
def run(bug_id_list,start=0,real_life = False,mutiple_bugs = False,mutiple_bugs_start=0):
    if real_life == 'True':
        group = real_life_bug_group
    else:
        group = bug_group
    interval = 50
    if mutiple_bugs == 'True':
        bug_id_list = random.sample(bug_id_list,5)
        start = mutiple_bugs_start
    else:
        start = group[bug_id_list[0]]['start']
    end = start + interval
    with open('/home/cedric/record/record_%d.txt'%0,'w') as f:
        f.write('bug id ')
        for id in bug_id_list:
            f.write(str(id) + ' ')
        f.write('\n')
        f.write('start : %d, end : %d'%(start,end))
    recoverAllFiles()
    inject_bugs(bug_id_list,real_life)
#     for i in bug_id_list:
#         os.system('cp /home/cedric/Desktop/copterTest/0/' + group[i]['file'] + ' /home/cedric/Desktop/arduPilot/' + group[i]['file'])
    os.chdir(dir_compile)
    os.system('bash bs.sh')
    time.sleep(15)
    if real_life and len(set([0,1,2]).intersection(set(bug_id_list))) != 0:
        run_sim_extend(start,end)
    else:
        run_sim(start,end)



if __name__ == '__main__':
#     run([0,1,7,11,15])
    real_life,mutiple_bugs = parserConfig()
    run(range(0,7),real_life=real_life,mutiple_bugs=mutiple_bugs,mutiple_bugs_start=0)
        # run([0,7,11,15,3])
        # run([2,5])
#     for i in range(0,18):
#         if i not in [0,1,7,11,15]:
#                 run([i])

#     run(range(9,18))
