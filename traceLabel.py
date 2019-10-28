import numpy as np
from numpy.linalg import inv
from numpy.linalg import pinv
from numpy.linalg import det
from numpy.linalg import cond
import math

n = 3
q = 3
p = 10
AR_dimension = 3
waypoint_num = 20

def simulationResultClean(cfg,index_from,index_to):
    dir = cfg.get('param','root_dir')+'experiment/output/PA/0/'
    states = []
    profiles = []
    for simulate_id in range(index_from,index_to+1):
        state = np.load(dir + 'states_np_%d_0.npy'%simulate_id)
        temp = []
        for s in state:
            temp.append([[x[0],x[1],x[2]] for x in s])
        profiles.append(np.load(dir + 'profiles_np_%d_0.npy'%simulate_id))
        states.append(temp)
    return states,profiles

def labelTraces_LR(states=None,profiles=None,std=6):
    true_labels = 0
    labels = []
    false_id = []
    for simulate_id,state in enumerate(states):
        profile = profiles[simulate_id]
        label = True
        for mission_id in range(0,len(profile)):
            if mission_id > 0 and mission_id % 3 == 0:
                continue
            state_temp = state[mission_id] # all the states for one mission
            profile_temp = profile[mission_id][:AR_dimension] # here we only focus on lat, lon, alt
            if not LinearRegressionBasedLabel(state_temp,profile_temp,std=std):
                label = False
                break
        if label:
            true_labels += 1
            labels.append(0)
        else:
            labels.append(1)
            false_id.append(simulate_id)
    print('total traces:%d'%len(states))
    print('positive labels:%d'%(len(states)-true_labels))
    return labels,false_id

def labelTraces_LR1(states=None,profiles=None,std=6):
    true_labels = 0
    labels = []
    false_id = []
    for simulate_id,state in enumerate(states):
        profile = profiles[simulate_id]
        label = True
        for mission_id in range(0,len(profile)):
            state_temp = state[mission_id]
            profile_temp = profile[mission_id][:AR_dimension]
            if not LinearRegressionBasedLabel(state_temp,profile_temp,std=std):
                label = False
                break
        if label:
            true_labels += 1
            labels.append(0)
        else:
            labels.append(1)
            false_id.append(simulate_id)
    print('total traces:%d'%len(states))
    print('positive labels:%d'%(len(states)-true_labels))
    return labels,false_id

def test_labelTraces(states=None,profiles=None,dir=None,start=0,end=1):
    true_labels = 0
    labels = []
    false_id = []
    false_labels_curve = 0
    false_labels_stable = 0
    for simulate_id,state in enumerate(states):
        curve_label = True
        stable_label = True
        profile = profiles[simulate_id]
        label = True
        for mission_id in range(0,len(profile)):
            state_temp = state[mission_id]
            profile_temp = profile[mission_id][:AR_dimension]
            if not LinearRegressionBasedLabel(state_temp,profile_temp,mission_id):
                label = False
                if mission_id > 0 and mission_id % 3 == 0:
                    curve_label = False
                else:
                    stable_label = False
        if label:
            true_labels += 1
            labels.append(0)
        else:
            labels.append(1)
            false_id.append(simulate_id)
        if not curve_label and stable_label:
            false_labels_curve += 1
        if not stable_label and curve_label:
            false_labels_stable += 1
    false_nums = len(states) - true_labels
    print('total traces:%d'%len(states))
    print('positive labels:%d'%false_nums)
    print('false_curve_labels:%d, proportion : %f'%(false_labels_curve,float(false_labels_curve)/false_nums))
    print('false_stable_labels:%d, proportion : %f'%(false_labels_stable,float(false_labels_stable)/false_nums))
    print('simultaneously false labels:%d, proportion : %f'%(false_nums - false_labels_curve-false_labels_stable,float(false_nums-false_labels_curve-false_labels_stable)/false_nums))

    return labels,false_id

def check_outlier_AR_based(errors,std=6):
    if len(errors) <= 4:
        return True
    ## dismiss the first prediction
    means = np.mean(errors[1:-1],axis=0)
    stds = np.std(errors[1:-1],axis=0)
    std_n = std
    for i in range(len(means)):
        if errors[-1][i] > means[i] + std_n * stds[i] or errors[-1][i] < means[i] - std_n * stds[i]:
            return False
    return True

def AR_based(state,profile=None):
    dimension = len(state[0])
    train,test = state[0:10],state[10:]
    models = [AR([x[i] for x in train]) for i in range(dimension)]
    model_fits = [models[i].fit() for i in range(dimension)]
    window = model_fits[0].k_ar
    coefs = [model_fits[i].params for i in range(dimension)]
    history = train[len(train)-window:]
    history = [history[i] for i in range(0,len(history))]
    predictions = []
    errors = []
    for t in range(len(test)):
        length = len(history)
        lag = [history[i] for i in range(length-window,length)]
        prediction = []
        error = []
        for dimension_id in range(dimension):
            yhat = coefs[dimension_id][0]
            for d in range(window):
                yhat += coefs[dimension_id][d+1] * lag[window-d-1][dimension_id]
            obs = test[t][dimension_id]
            prediction.append(yhat)
            error.append(yhat - obs)
        predictions.append(prediction)
        errors.append(error)
        history.append(test[t])
        
        if not check_outlier_AR_based(errors):
            return False
    return True

def computeYk(k,state,U):
    Yk = state[k-1]
    for index in range(k-2,k-p-1,-1):
        Yk = np.append(Yk,state[index])
    Yk = np.append(Yk,U)
    Yk = np.transpose(Yk)
    return Yk.reshape(n*p+q,1)

def LinearRegressionBasedLabel(state,profile,std=6):
    state = np.array(state)
    labels = []
    lambd = 0.95
    U = profile
    # X_i = state[p].reshape(n,1)
    Yk = computeYk(p,state,U)
    YkT = np.transpose(Yk)
    # Y_i = Yk

    psi_i = np.ones((n,n*p+q))
    phi_i = np.identity(n*p+q)
    phi_i_inv = inv(phi_i)
    A_i = np.mat(psi_i)*np.mat(phi_i_inv)
    errors = []
    i = p
    while i < len(state):
        X_est = np.mat(A_i)*np.mat(Yk)
        errors.append(X_est.A1 - state[i])
        if not check_outlier_AR_based(errors,std):
            return False
        ## start to predict state[i+1], so we need training data up to state[i]
        Yk = computeYk(i+1,state,U) # up to state[i]
        YkT = np.transpose(Yk)
        # X_i = np.concatenate((state[i].reshape(n,1),math.sqrt(lambd)*X_i),axis=1)
        # Y_i = np.concatenate((Yk,math.sqrt(lambd)*Y_i),axis=1)
        psi_i = lambd*psi_i + np.mat(state[i].reshape(n,1))*np.mat(YkT)
        phi_i = lambd*phi_i + np.mat(Yk)*np.mat(YkT)
        phi_i_inv = phi_i_inv / lambd - math.pow(lambd,-2)*np.mat(phi_i_inv)*np.mat(Yk)*inv(np.mat(np.identity(1)+np.mat(YkT)*np.mat(phi_i_inv)*np.mat(Yk)/lambd))*np.mat(YkT)*np.mat(phi_i_inv)
        A_i = np.mat(psi_i)*np.mat(phi_i_inv)
        i += 1
    print('---------------')
    print(errors)
    return True

if __name__ == '__main__':
    start = 0
    end = start
    states,profiles = simulationResultClean(start,end)
    labelTraces_LR(states,profiles)
    exit(0)


