import os
from injector import *
import time
from runsimulation import *
from runsimul_extend import run_sim_extend
import random
from ConfigParser import ConfigParser

def parserConfig():
    cfg = ConfigParser()
    cfg.read('config.ini')
    config = {}
    config['root_dir'] = cfg.get('param','root_dir')
    config['real_life'] = cfg.get('param','real_life')
    config['mutiple_bugs'] = cfg.get('param','mutiple_bugs')
    config['start'] = int(cfg.get('param','start'))
    config['end'] = int(cfg.get('param','end'))
    config['rounds'] = int(cfg.get('param','rounds'))
    return config

def recoverAllFiles():
    for bug in bug_group:
        os.system('cp ' + config['root_dir'] + '.ArduPilot_Back/' + bug['file'] + ' ' + config['root_dir'] + bug['file'])
    for bug in real_life_bug_group:
        os.system('cp ' + config['root_dir'] + '.ArduPilot_Back/' + bug['file'] + ' ' + config['root_dir'] + bug['file'])
        
def run(config):
    if config['real_life'] == 'True':
        group = real_life_bug_group
    else:
        group = bug_group
    if config['mutiple_bugs'] == 'True':
        bug_id_list = random.sample(list(range(0,len(group))),5)
    else:
        bug_id_list = random.sample(list(range(0,len(group))),1)
    print(bug_id_list)
    start = config['start']
    end = config['end']
    with open('%s/record_start=%d.txt'%(config['root_dir']+'experiment',config['start']),'w') as f:
        f.write('real_life : ' + config['real_life'] + '\n')
        f.write('mutiple_bugs : ' + config['mutiple_bugs'] + '\n')
        f.write('bug id : ')
        for id in bug_id_list:
            f.write(str(id) + ' ')
        f.write('\n')
        f.write('start : %d, end : %d'%(start,end))
    recoverAllFiles()
    inject_bugs(bug_id_list,config)
#     for i in bug_id_list:
#         os.system('cp /home/cedric/Desktop/copterTest/0/' + group[i]['file'] + ' /home/cedric/Desktop/arduPilot/' + group[i]['file'])
    os.chdir(config['root_dir'])
    os.system('make sitl -j4')
    time.sleep(15)
    os.system('cp build/sitl/bin/arducopter experiment/elf/0/ArduCopter.elf')
    time.sleep(3)
    if config['real_life'] and len(set([0,1,2]).intersection(set(bug_id_list))) != 0:
        run_sim_extend(config)
    else:
        run_sim(config)



if __name__ == '__main__':
    config = parserConfig()
    interval = config['end'] - config['start']
    for i in range(0,config['rounds']):
        config['start'] += i * interval
        config['end'] += i * interval
        run(config)