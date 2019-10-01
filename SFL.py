from traceLabel import *
from injector import *
from ConfigParser import ConfigParser

def statics(bug_id_list,group,cfg):
    all_lines = set()
    for bug_id in bug_id_list:
        with open(cfg.get('param','root_dir')+group[bug_id]['file'],'r') as f:
            for line_no,line in enumerate(f,1):
                if 'EXECUTE_MARK()' in line:
                    all_lines.add(group[bug_id]['file']+'-'+str(line_no))
    return list(all_lines)

def executionTracesClean(bug_id_list,group,start,end,cfg):
    output_dir = cfg.get('param','root_dir')+'experiment/output/PA/0/'
    traces = []
    for simulate_id in range(start,end+1):
        # trace_set = set()
        current_trace = []
        with open(output_dir+'raw_%s_0.txt'%simulate_id) as f:
            for line in f:
                for bug_id in bug_id_list:
                    if group[bug_id]['file'] in line:
                        temp_str = line.split('-')[2].strip()
                        if ':' in temp_str: # the record of sum of execution times
                            # todo : take advantage of the sum times
                            continue
                        new_line = group[bug_id]['file']+'-'+str(temp_str)
                        # if new_line not in trace_set:
                        current_trace.append(new_line)
                            # trace_set.add(new_line)
                        break
                    # current_pos_trace.append(int(temp_str[:temp_str.find(':')]))
        current_trace.sort()
        if len(current_trace) == 0:
            print('log execution traces fail for simulate_id %d'%simulate_id)
        traces.append(current_trace)
    return traces
    
def compressSameValue(suspicious):
    sus = []
    for element in suspicious:
        if len(sus) == 0 or element[0] != sus[-1][0]:
            sus.append([element[0],[element[1]]])
        else:
            sus[-1][1].append(element[1])
    return sus
def tarantula(all_lines,traces,labels):
    suspicious = []
    total_failed = sum(labels)
    if total_failed == 0:
        print('no positive labels')
        return 
    total_passed = len(labels) - total_failed
    if total_passed == 0:
        print('no negative labels')
        return
    for line in all_lines:
        passed = 0.0
        failed = 0.0
        for trace_id,trace in enumerate(traces):
            if line in trace:
                if labels[trace_id] == 0: #passed
                    passed += 1
                else:
                    failed += 1
        if passed + failed == 0:
            suspici = 0
        else:
            suspici = failed / total_failed / (passed/total_passed+failed/total_failed)
        suspicious.append([suspici,line])
    suspicious.sort(reverse=True)
    return suspicious
    #return compressSameValue(suspicious)
    

def crosstab(all_lines,traces,labels):
    suspicious = []
    total_num = len(labels)
    total_failed = sum(labels)
    total_passed = total_num - total_failed
    for line in all_lines:
        ncs = 0.0
        ncf = 0.0
        nus = 0.0
        nuf = 0.0
        for trace_id,trace in enumerate(traces):
            if line in trace:
                if labels[trace_id] == 0:
                    ncs += 1
                else:
                    ncf += 1
            else:
                if labels[trace_id] == 0:
                    nus += 1
                else:
                    nuf += 1
        ns = ncs + nus
        nf = ncf + nuf
        nc = ncs + ncf
        nu = nus + nuf
        ecf = (nc * total_failed) / total_num
        ecs = (nc * total_passed) / total_num
        euf = (nu * total_failed) / total_num
        eus = (nu * total_passed) / total_num
        # print(ecf)
        # print(ecs)
        # print(euf)
        # print(eus)
        if ecf == 0:
            t1 = 0
        else:
            t1 = math.pow(ncf-ecf,2)/ecf
        if ecs == 0:
            t2 = 0
        else:
            t2 = math.pow(ncs-ecs,2)/ecs
        if euf == 0:
            t3 = 0
        else:
            t3 = math.pow(nuf-euf,2)/euf
        if eus == 0:
            t4 = 0
        else:
            t4 = math.pow(nus-eus,2)/eus
        chi_square = t1 + t2 + t3 + t4
        M = chi_square / total_num
        if nf != 0 and ncs != 0:
            phi = ncf / nf / ncs * ns
        else:
            phi = 0
        if phi > 1:
            suspicious.append([M,line])
        elif phi == 1:
            suspicious.append([0,line])
        else:
            suspicious.append([-M,line])
    suspicious.sort(reverse=True)
    return suspicious
    #return compressSameValue(suspicious)
    
def print_line_info(all_lines,traces,lineno):
    result = []
    positive_traces_id = []
    for line_no in all_lines:
        temp = 0
        l = []
        for trace_id,trace in enumerate(traces):
            if line_no in trace:
                temp += 1
                if line_no in lineno:
                    positive_traces_id.append(trace_id)
                l.append(trace_id)
        # result.append([line_no,temp])
        result.append([temp,line_no])
        if line_no in lineno:
            print([line_no,temp])
    print(set(positive_traces_id))
    result.sort(reverse=True)
    print(result)

def sus_analysis(lines,sus_list):
    for sus in sus_list:
        if sus == None:
            continue
        for i in range(0,len(sus)):
            for line in lines:
                if line in sus[i][1]:
                    print('line no %s rank #%d sus : %f'%(line,i,sus[i][0]))
    print('~~~~~~~~~~~')

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

def analysis(cfg,bug_id_list):
    if cfg.get('param','real_life') == 'True':
        group = real_life_bug_group
    else:
        group = bug_group
    # bug_id_list = [0,1,7,11,15]
    all_lines = statics(bug_id_list,group,cfg)
    # print(all_lines)
    start = int(cfg.get('param','start'))
    ##### test std : end = start + 200
    # end = int(cfg.get('param','end'))
    end = start + 200
    traces = executionTracesClean(bug_id_list,group,start,end-1,cfg)
    # print(all_lines)
    print(len(traces))
    for bug_id in bug_id_list:
        print(str(bug_id) + ':' + group[bug_id]['file'])
    states,profiles = simulationResultClean(cfg,start,end-1)
    # labels,positives = test_labelTraces(states,profiles)
    # return
    labels1,positive1 = labelTraces_LR(states,profiles)
    labels2,positive2 = labelTraces_LR1(states,profiles)
    positive_id = set()
    for i in range(0,len(traces)):
        for bug_id in bug_id_list:
            for line_no in group[bug_id]['lineno']:
                line = group[bug_id]['file'] + '-' + str(line_no)
                if line in traces[i] :
                    positive_id.add(i)
                    break
    positive_id = list(positive_id)
    print('ground truth : ' + str(len(positive_id)))
    negative_id = []
    for i in range(0,len(traces)):
        if i not in positive_id:
            negative_id.append(i)
    false_positive1 = 0
    for id in positive1:
        if id in negative_id:
            false_positive1 += 1
    false_positive2 = 0
    for id in positive2:
        if id in negative_id:
            false_positive2 += 1
    false_negative1 = 0
    for i in range(0,len(traces)):
        if i not in positive1 and i in positive_id:
            false_negative1 += 1
    false_negative2 = 0
    for i in range(0,len(traces)):
        if i not in positive2 and i in positive_id:
            false_negative2 += 1

    # print('false1_positive : %d / %d'%(false_positive1,len(negative_id)))
    # print('false2_positive : %d / %d'%(false_positive2,len(negative_id)))
    # print('false1_negative : %d / %d'%(false_negative1,len(positive_id)))
    # print('false2_negative : %d / %d'%(false_negative2,len(positive_id)))
    if len(negative_id) != 0:
        print('false positive rate1 : %f'%(float(false_positive1)/len(negative_id)))
        # print('false positive rate2 : %f'%(float(false_positive2)/len(negative_id)))
    else:
        print('false positive rate1 : None')
        # print('false positive rate2 : None')
    if len(positive_id) != 0:
        print('false negative rate1 : %f'%(float(false_negative1)/len(positive_id)))
        # print('false negative rate2 : %f'%(float(false_negative2)/len(positive_id)))
    else:
        print('false negative rate1 : None')
        # print('false negative rate2 : None')
    sus_tar1 = tarantula(all_lines,traces,labels1)
    # sus_tar2 = tarantula(all_lines,traces,labels2)
    sus_cro1 = crosstab(all_lines,traces,labels1)
    # sus_cro2 = crosstab(all_lines,traces,labels2)

    for bug_id in bug_id_list:
        lines = []
        for line_no in group[bug_id]['lineno']:
            lines.append(group[bug_id]['file']+'-'+str(line_no))
        print(lines)
        # sus_analysis(lines,[sus_tar1,sus_tar2,sus_cro1,sus_cro2]) 
        sus_analysis(lines,[sus_tar1,sus_cro1]) 

def mainRecord(config):
    record_path = config['root_dir'] + 'experiment/'
    record_files = [f for f in os.listdir(record_path) if f.startswith('start') ]
    print(record_files)
    for record_file in record_files:
        print(record_file)
        cfg = ConfigParser()
        cfg.read(record_path+record_file)
        temp = cfg.get('param','bug')[1:-1]
        bug_id_list = [int(t.strip()) for t in temp.split(',')]
        analysis(cfg,bug_id_list)

if __name__ == '__main__':
    config = parserConfig()
    mainRecord(config)
    
        
