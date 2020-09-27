import os
from injector import *
import time
from runsimulation import *
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

def writeConfig(cfg_name,bug_id_list,start,end):
    cfg = ConfigParser()
    cfg.read('monkey-copter/config.ini')
    cfg.set('param','bug',str(bug_id_list))
    cfg.set('param','start',str(start))
    cfg.set('param','end',str(end))
    cfg.write(open('experiment/'+cfg_name,'w'))

def recoverAllFiles():
    for bug in bug_group:
        os.system('cp ' + config['root_dir'] + '.ArduPilot_Back/' + bug['file'] + ' ' + config['root_dir'] + bug['file'])
    for bug in real_life_bug_group:
        os.system('cp ' + config['root_dir'] + '.ArduPilot_Back/' + bug['file'] + ' ' + config['root_dir'] + bug['file'])
        
def run(config):
    if config['real_life'] == 'True':
        if config['mutiple_bugs'] == 'True':
            group = list(range(0,len(real_life_bug_group)))
        else:
            group = [3,4,5,6,7]
    else:
        group = [0,1,3,6,10,11,12,14]
        # group = bug_group
    if config['mutiple_bugs'] == 'True':
        bug_id_list = random.sample(group,5)
        # bug_id_list = random.sample(list(range(0,len(group))),5)
    else:
        bug_id_list = random.sample(group,1)
        # bug_id_list = random.sample(list(range(0,len(group))),1)
    print(bug_id_list)
    #bug_id_list = [9]
    start = config['start']
    end = config['end']
    recoverAllFiles()
    inject_bugs(bug_id_list,config)
    os.chdir(config['root_dir'])
    cfg_name = 'foo_'+str(start)+'.ini'
    os.system('cp monkey-copter/config.ini experiment/'+cfg_name)
    writeConfig(cfg_name,bug_id_list,start,end)
    os.system('make sitl -j4')
    time.sleep(15)
    os.system('cp build/sitl/bin/arducopter experiment/elf/0/ArduCopter.elf')
    time.sleep(3)
    if config['real_life'] == 'True' and len(set([0]).intersection(set(bug_id_list))) != 0:
        run_sim(config,1)
    else:
        run_sim(config,0)

if __name__ == '__main__':
    config = parserConfig()
    interval = config['end'] - config['start']
    for i in range(0,config['rounds']):
        run(config)
        config['start'] += interval
        config['end'] += interval
