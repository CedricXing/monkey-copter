from traceLabel import *
from injector import *

dir = '/home/cedric/Desktop/arduPilot/'

def statics(bug_id_list,group):
    all_lines = set()
    for bug_id in bug_id_list:
        with open(dir+group[bug_id]['file'],'r') as f:
            for line_no,line in enumerate(f,1):
                if 'EXECUTE_MARK()' in line:
                    all_lines.add(group[bug_id]['file']+'-'+str(line_no))
    return list(all_lines)

def executionTracesClean(bug_id_list,group,start=0,end=0):
    output_dir = '/home/cedric/Desktop/arduPilot/experiment/output/PA/0/'
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
    return compressSameValue(suspicious)
    

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
    return compressSameValue(suspicious)
    
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

if __name__ == '__main__':
    interval = 49
    # group = real_life_bug_group
    group = bug_group
    # bug_id_list = [0,1,7,11,15]
    bug_id_list = [1,2,6,8,10]
    all_lines = statics(bug_id_list,group)
    # print(all_lines)
    start = 0
    traces = executionTracesClean(bug_id_list,group,start=0,end=start+interval)
    # print(all_lines)
    print(len(traces))
    for bug_id in bug_id_list:
        print(str(bug_id) + ':' + group[bug_id]['file'])
    states,profiles = simulationResultClean(start,start+interval)
    labels1,positive1 = labelTraces_LR(states,profiles)
    labels2,positive2 = labelTraces_LR1(states,profiles)
    positive_id = []
    for i in range(0,len(traces)):
        for bug_id in bug_id_list:
            for line_no in group[bug_id]['lineno']:
                if line_no in traces[i] :
                    positive_id.append(i)
                    break
    print(positive_id)
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

        # print('false1_positive : %d / %d = %f'%(false_positive1,len(negative_id),float(false_positive1)/len(negative_id)))
        # print('false2_positive : %d / %d = %f'%(false_positive2,len(negative_id),float(false_positive2)/len(negative_id)))
        # print('false1_negative : %d / %d = %f'%(false_negative1,len(positive_id),float(false_negative1)/len(positive_id)))
        # print('false2_negative : %d / %d = %f'%(false_negative2,len(positive_id),float(false_negative2)/len(positive_id)))
    print('false1_positive : %d / %d'%(false_positive1,len(negative_id)))
    print('false2_positive : %d / %d'%(false_positive2,len(negative_id)))
    print('false1_negative : %d / %d'%(false_negative1,len(positive_id)))
    print('false2_negative : %d / %d'%(false_negative2,len(positive_id)))

    sus_tar1 = tarantula(all_lines,traces,labels1)
    sus_cro1 = crosstab(all_lines,traces,labels1)
    sus_tar2 = tarantula(all_lines,traces,labels2)
    sus_cro2 = crosstab(all_lines,traces,labels2)
    print(sus_tar1)
    # print(sus_cro2)
    # print(sus_tar2)
    # print(sus_cro2)
    for bug_id in bug_id_list:
        lines = []
        for line_no in group[bug_id]['lineno']:
            lines.append(group[bug_id]['file']+'-'+str(line_no))
        print(lines)
        sus_analysis(lines,[sus_tar1,sus_tar2,sus_cro1,sus_cro2]) 
    # sus_analysis(bug['lineno'],[sus_tar1,sus_tar2,sus_cro1,sus_cro2])
    # print_line_info(all_lines,traces,bug['lineno'])
    # print('-----------------------------------------')
    
    # for bug_id in bug_id_list:
    #     bug = group[bug_id]
    #     all_lines = statics(bug['file'])
    #     print('bug id : %d'%bug_id)
    #     print('bug file : %s'%bug['file'])
    #     start = bug['start']
    #     states,profiles = simulationResultClean(start,start+interval)
    #     # states,profiles = simulationTiny(start,start+interval)
    #     traces = executionTracesClean(bug['file'],start=start,end=start+interval)
    #     # labels1,false1 = labelTraces_LR(start=start,end=end)
    #     # labels2,false2 = labelTraces_LR1(start=start,end=end)
    #     labels1,positive1 = labelTraces_LR(states,profiles)
    #     labels2,positive2 = labelTraces_LR1(states,profiles)
        
