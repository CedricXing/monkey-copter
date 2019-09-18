import numpy as np
from numpy.linalg import inv
from numpy.linalg import pinv
from numpy.linalg import det
from numpy.linalg import cond
import math
# from statsmodels.tsa.ar_model import AR


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
        # print(state)
        temp = []
        for s in state:
            temp.append([[x[0],x[1],x[2]] for x in s])
        profiles.append(np.load(dir + 'profiles_np_%d_0.npy'%simulate_id))
        states.append(temp)
    # for i in range(len(states)):
        # for j in range(len(states[i])):
            # print(states[i][j])
            # states[i][j] = states[i][j] + np.random.random((len(states[i][j]),len(states[i][j][0]))) / 10000000
            # print(states[i][j])
            # break
        # break
        # exit(0)
    # print(states[0][0])
    # exit(0)
    return states,profiles

def simulationTiny(index_from,index_to):
    dir = '/home/cedric/ArduPilot/experiment/output/PA/0/'
    states = []
    profiles = []
    for simulate_id in range(index_from,index_to+1):
        states_temp = []
        profiles.append(np.load(dir+'profiles_np_%d_0.npy'%simulate_id))
        with open(dir+'raw_%d_0.txt'%index_from,'r') as f:
            lines = f.readlines()
            for line in lines:
                para = []
                if 'lat' in line and 'lon' in line:
                    strings = line.split(':')
                    para.append(float(strings[1].strip())/10000000)
                    para.append(float(strings[3].strip())/10000000)
                    para.append(float(strings[5].strip())/100)
                    states_temp.append(para)
        states_new = []
        length = len(states_temp)
        for i in range(length-waypoint_num*40*4,length,40):
            # print(np.array(states_temp[i:i+40]))
            states_new.append(states_temp[i:i+40])
        states.append(states_new)
    # print(states[0][0][0])
    # print(profiles[0][0])

    return states,profiles

def simulationResultClean_LR(dir,index_from,index_to):
    states = []
    profiles = []
    for simulate_id in range(index_from,index_to+1):
        with open(dir + 'profiles_%d.pckl'%simulate_id) as f:
            p = pickle.load(f)[0]
            profiles.append([[p.lat,p.lon,p.alt+20],[p.target1.lat,p.target1.lon,p.target1.alt],[p.target2.lat,p.target2.lon,p.target2.alt],
            [p.target3.lat,p.target3.lon,p.target3.alt],[p.target4.lat,p.target4.lon,p.target4.alt]])
        states.append(np.load(dir + 'states_np_%d_0.npy'%simulate_id))
    return states,profiles

def get_distance_metres(aLocation1, aLocation2): ### Cedric: it will not be arrurate over large distance and close to the earth's poles.
    # aLocation1 = LocationGlobal(aLocation1)
    # aLocation2 = LocationGlobal(aLocation2) 
    dlat = aLocation2[0] - aLocation1[0]
    dlong = aLocation2[1] - aLocation1[1]
    return math.sqrt((dlat*dlat) + (dlong*dlong)) * 1.113195e5

def compute_divergence(traces):
    divergences = []
    steps = len(traces[0])
    for i in range(0,steps):
        divergence = []
        divergence.append(get_distance_metres(traces[0][i],traces[1][i]))
        divergence.append(get_distance_metres(traces[0][i],traces[2][i]))
        divergence.append(get_distance_metres(traces[0][i],traces[3][i]))
        divergence.append(get_distance_metres(traces[1][i],traces[2][i]))
        divergence.append(get_distance_metres(traces[1][i],traces[3][i]))
        divergence.append(get_distance_metres(traces[2][i],traces[3][i]))
        divergence.sort()
        divergences.append(divergence)
    return divergences

def labelTraces_LR(states=None,profiles=None,dir=None,start=0,end=1):
    # dir = '/home/cedric/ArduPilot/experiment/output/PA/0/'
    # states,profiles = simulationResultClean(dir,start,end)
    true_labels = 0
    labels = []
    false_id = []
    for simulate_id,state in enumerate(states):
        profile = profiles[simulate_id]
        # print(profile)
        label = True
        for mission_id in range(0,len(profile)):
            if mission_id > 0 and mission_id % 3 == 0:
                # print('mission id :'+str(mission_id))
                continue
            # print(mission_id)
            # print(simulate_id)
            state_temp = state[mission_id]
            # print(len(state_temp))
            # exit(0)
            profile_temp = profile[mission_id][:AR_dimension]
            # print(state_temp)
            # print(profile_temp)
            if not LinearRegressionBasedLabel(state_temp,profile_temp,mission_id):
                label = False
                # print(mission_id)
                # print(simulate_id)
                break
        if label:
            true_labels += 1
            labels.append(0)
        else:
            # print(simulate_id)
            labels.append(1)
            false_id.append(simulate_id)
            # break
        # break
        # exit(0)
    print('total traces:%d'%len(states))
    print('positive labels:%d'%(len(states)-true_labels))
    # print('true labels:%d'%true_labels)
    print(false_id)
    return labels,false_id

def labelTraces_LR1(states=None,profiles=None,dir=None,start=0,end=1):
    # dir = '/home/cedric/ArduPilot/experiment/output/PA/0/'
    # states,profiles = simulationResultClean(dir,start,end)
    true_labels = 0
    labels = []
    false_id = []
    for simulate_id,state in enumerate(states):
        profile = profiles[simulate_id]
        label = True
        for mission_id in range(0,len(profile)):
            state_temp = state[mission_id]
            # print(state_temp)
            # exit(0)
            profile_temp = profile[mission_id][:AR_dimension]
            if not LinearRegressionBasedLabel(state_temp,profile_temp):
            # if not AR_based(state_temp,profile_temp):
                label = False
                # print(mission_id)
                # print(simulate_id)
                break
        if label:
            true_labels += 1
            labels.append(0)
        else:
            # print(simulate_id)
            labels.append(1)
            false_id.append(simulate_id)
            # break
        # break
    print('total traces:%d'%len(states))
    # print('true labels:%d'%true_labels)
    print('positive labels:%d'%(len(states)-true_labels))
    print(false_id)
    return labels,false_id

# def labelTraces_LR1(states=None,profiles=None,dir=None,start=0,end=1):
#     # dir = '/home/cedric/ArduPilot/experiment/output/PA/0/'
#     # states,profiles = simulationResultClean(dir,start,end)
#     true_labels = 0
#     labels = []
#     false_id = []
#     for simulate_id,state in enumerate(states):
#         states_all = []
#         profile_all = []
#         profile = profiles[simulate_id][0]
#         label = True
#         # print(len(profile))
#         for mission_id in range(0,len(profile)):
#             if mission_id == 0:
#                 continue
#             state_temp = state[mission_id]
#             profile_temp = profile[mission_id][:3]
#             for i in range(0,len(state_temp)):
#                 states_all.append(state_temp[i])
#                 profile_all.append(profile_temp)
#         # print(states_all)
#         # print(profile_all)
#         # exit(0)
#         if not LinearRegressionBasedLabel1(states_all,profile_all):
#             label = False
#             labels.append(1)
#             false_id.append(simulate_id)
#         else:
#             true_labels += 1
#             labels.append(0)
#         # break
#     print('total traces:%d'%len(states))
#     print('true labels:%d'%true_labels)
#     print(false_id)
#     return labels,false_id

def labelTraces_AR(states=None,profiles=None,dir=None,start=0,end=1):
    # states,profiles = simulationResultClean(dir,start,end)
    true_labels = 0
    labels = []
    false_id = []
    for simulate_id,state in enumerate(states):
        # print(state)
        # exit(0)
        # if simulate_id == 0 or simulate_id == 1:
        #     continue
        # print('simulate id : %d'%simulate_id)
        # print(state)
        # if simulate_id == 0:
        #     continue
        profile = profiles[simulate_id][0]
        label = True
        # print(state[1])
        # exit(0)
        # print(len(profile))
        for mission_id in range(0,len(profile)):
            if mission_id <= 1:
                continue
            # print(mission_id)
            # state_temp = np.transpose(np.transpose(state[mission_id])[:AR_dimension])
            state_temp = state[mission_id]
            # print(len(state_temp))
            # exit(0)
            profile_temp = profile[mission_id][:AR_dimension]
            # print(state_temp[18:])
            if not AR_based(state_temp[18:],profile_temp):
                label = False
                # print('mission id : %d'%mission_id)
                # print('-------------------------------------------------')
                break
        if label:
            true_labels += 1
            labels.append(0)
        else:
            false_id.append(simulate_id)
            labels.append(1)
            # break
        # exit(0)
        # break
        # print('-----------------------------')
    print('total traces:%d'%len(states))
    print('true labels:%d'%true_labels)
    print(false_id)
    return labels,false_id

def labelTraces_AR1(states=None,profiles=None,dir=None,start=0,end=1):
    # dir = '/home/cedric/ArduPilot/experiment/output/PA/0/'
    # states,profiles = simulationResultClean(dir,start,end)
    true_labels = 0
    labels = []
    false_id = []
    for simulate_id,state in enumerate(states):
        # print('simulate id : %d'%simulate_id)
        profile = profiles[simulate_id][0]
        label = True
        last_state = state[0]
        for mission_id in range(0,len(profile)):
            # print(mission_id)
            current_state = state[mission_id]
            # state_temp = np.concatenate(np.array(last_state[-7:,]),np.array(current_state))
            state_temp = last_state[-7:]
            # state_temp = state_temp.tolist()
            # print(state_temp.tolist())
            # print(state_temp)
            state_temp = np.concatenate((state_temp,current_state))
            # print(state_temp)
            # exit(0)
            # for s in current_state:
                # state_temp.append(s)
            last_state = current_state
            if mission_id <=1:
                continue
            # print(state_temp)
            # exit(0)
            # print(np.array(state_temp[18:]))
            profile_temp = profile[mission_id][:AR_dimension]
            # print(state_temp[25:])
            if not AR_based(state_temp[0:25],profile_temp) or not AR_based(state_temp[25:],profile_temp):
                label = False
                # print('mission id : %d'%mission_id)
                # print('-------------------------------------------------')
                break
        if label:
            true_labels += 1
            labels.append(0)
        else:
            false_id.append(simulate_id)
            labels.append(1)
            # break
        # exit(0)
        # break
        # print('-----------------------------')
    print('total traces:%d'%len(states))
    print('true labels:%d'%true_labels)
    print(false_id)
    return labels,false_id

def check_outlier_AR_based(errors):
    if len(errors) <= 4:
        return True
    # print(len(errors))
    means = np.mean(errors[1:-1],axis=0)
    stds = np.std(errors[1:-1],axis=0)
    std_n = 6
    # print(stds)
    for i in range(len(means)):
        if errors[-1][i] > means[i] + std_n * stds[i] or errors[-1][i] < means[i] - std_n * stds[i]:
            # print('dimension %d'%i)
            # print(len(errors))
            # print(errors[20:-1])
            # print(means)
            # print(means-std_n*stds)
            # print(errors[-1])
            # print(means+std_n*stds)
            # print('--------------------------------------------------')
            return False
    return True

def AR_based(state,profile=None):
    # noises = np.random.random((len(state),len(state[0]))) / 100000000
    # print(state)
    # noises = np.zeros((len(state),len(state[0])))
    # state = state + noises
    # print(state)
    # print(state)
    # exit(0)
    # print(state[:,2:len(state[0])])
    # state[:,2:len(state[0])] += np.random.random((len(state),len(state[0])-2))
    # print(state[:,2:len(state[0])])
    # print(state[30:38])
    # exit(0)
    dimension = len(state[0])
    train,test = state[0:10],state[10:]
    # print([x[0] for x in train])
    models = [AR([x[i] for x in train]) for i in range(dimension)]
    # for model in models:
    #     model.add_trend(has_constant='skip')
    # print(train)
    model_fits = [models[i].fit() for i in range(dimension)]
    window = model_fits[0].k_ar
    # print(window)
    # print('hello')
    # for model in model_fits:
    #     print(model.k_ar)
    coefs = [model_fits[i].params for i in range(dimension)]
    # print(len(coefs[0]))
    # exit(0)
    # print(coef)
    history = train[len(train)-window:]
    # print(history)
    history = [history[i] for i in range(0,len(history))]
    predictions = []
    errors = []
    for t in range(len(test)):
        length = len(history)
        lag = [history[i] for i in range(length-window,length)]
        prediction = []
        error = []
        for dimension_id in range(dimension):
            # print(dimension_id)
            yhat = coefs[dimension_id][0]
            for d in range(window):
                # print(d)
                yhat += coefs[dimension_id][d+1] * lag[window-d-1][dimension_id]
            obs = test[t][dimension_id]
            prediction.append(yhat)
            error.append(yhat - obs)
        # print(prediction)
        # print(error)
        predictions.append(prediction)
        errors.append(error)
        history.append(test[t])
        
        # print(predictions[-1][2])
        if not check_outlier_AR_based(errors):
            # print('train')
            # print(train)
            # print('test')
            # print(test)
            # # print('-----------------------------')
            # print(t)
            # print(test[t-1])
            # print(test[t])
            # print(prediction)
            # print(predictions)
            # print(predictions[-2])
            # for pos,data in enumerate(test):
            #     print(pos)
            #     print(data)
            # print(prediction)
            # print(train)
            # exit(0)
            # if len(state) == 15:
            #     print('--------------------------------------------------------------------')
            return False
        # print('predicted=%f, expected=%f'%(yhat,obs))
    return True

def AR_based1(state,profile=None):
    # print(state)
    noises = np.random.random((len(state),len(state[0]))) / 10000000
    state = state + noises
    # print(state[:,2:len(state[0])])
    # state[:,2:len(state[0])] += np.random.random((len(state),len(state[0])-2))
    # print(state[:,2:len(state[0])])
    # print(state[30:38])
    # exit(0)
    dimension = len(state[0])
    # print(state)
    # exit(0)
    train,test = state[5:10],state[10:]
    # print(train)
    # exit(0)
    # models = [AR(train[:,i]) for i in range(dimension)]
    models = [AR([x[i] for x in train]) for i in range(dimension)]
    model_fits = [models[i].fit(maxlag=4) for i in range(dimension)]
    window = 5
    coefs = [model_fits[i].params for i in range(dimension)]
    # print(coef)
    history = train[len(train)-window:]
    # print(history)
    history = [history[i] for i in range(0,len(history))]
    length = len(history)
    lag = [history[i] for i in range(length-window,length)]
    predictions = []
    errors = []
    for t in range(len(test)):
        prediction = []
        error = []
        for dimension_id in range(dimension):
            pred = model_fits[dimension_id].predict(start=window,end=len(state))[0]
            print(pred)
            obs = test[t][dimension_id]
            prediction.append(pred)
            error.append(pred - obs)
        predictions.append(prediction)
        errors.append(error)
        history.append(test[t])
        # print(predictions[-1][2])
        if not check_outlier_AR_based(errors):
            return False
        length = len(history)
        lag = [history[i] for i in range(length-window,length)]
        # print(len(lag))
        # print(window)
        # temp = []
        # for x in lag:
        #     temp.append(x[0])
        # print(temp)
        models = [AR([x[i] for x in lag]) for i in range(dimension)]
        model_fits = [models[i].fit(maxlag=4) for i in range(dimension)]
        coefs = [model_fits[i].params for i in range(dimension)]
        # print('predicted=%f, expected=%f'%(yhat,obs))
    return True

def computeYk(k,state,U):
    Yk = state[k-1]
    for index in range(k-2,k-p-1,-1):
        # print(index)
        Yk = np.append(Yk,state[index])
    Yk = np.append(Yk,U)
    Yk = np.transpose(Yk)
    return Yk.reshape(n*p+q,1)

def check_outlier(X_est,state):
    divergence = X_est.A1 - state
    # print(X_est.A1)
    # print(state)
    # print(divergence)
    # print('--------------------------------------------')
    
    if abs(divergence[0]) > 0.1 or abs(divergence[1] > 0.1) or abs(divergence[2]) > 5:
        return False
    return True

def check_outlier_new(X_est,state,errors):
    # divergence = state - X_est.A1
    divergence = X_est.A1 - state
    # print(state)
    # print(X_est.A1)
    if len(errors) == 0:
        errors.append(divergence)
        return True
    # print(X_est.A1)
    # print(state)
    std_n = 6
    errors.append(divergence)
    # print(errors)
    means = np.mean(errors,axis=0)
    stds = np.std(errors,axis=0)
    # print(means)
    # print(stds)
    # errors.append(divergence)
    # print(means-std_n*stds)
    # print(divergence)
    # print(means+std_n*stds)
    # print('--------------------------------------------')
    if divergence[0] > means[0] + std_n*stds[0] or divergence[0] < means[0] - std_n*stds[0] or divergence[1] > means[1] + std_n*stds[1] or divergence[1] < means[1] - std_n*stds[1] or divergence[2] > means[2] + std_n*stds[2] or divergence[2] < means[2] - std_n*stds[2]:
        # print(means-std_n*stds)
        # print(divergence)
        # print(means+std_n*stds)
        # print('--------------------------------------------')
        return False
    return True

def LinearRegressionBasedLabel(state,profile,mission_id=0):
    state = np.array(state)
    # print('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')
    labels = []
    i = p
    lambd = 0.95
    U = profile
    X_i = state[p].reshape(n,1)
    Yk = computeYk(p,state,U)
    Y_i = Yk

    psi_i = np.ones((n,n*p+q))
    phi_i = np.identity(n*p+q)
    phi_i_inv = inv(phi_i)
    A_i = np.mat(psi_i)*np.mat(phi_i_inv)
    errors = []
    flag = True
    while i < len(state):
        X_est = np.mat(A_i)*np.mat(Yk)
        errors.append(X_est.A1 - state[i])
        Yk = computeYk(i,state,U)
        YkT = np.transpose(Yk)
        # print(X_est.A1)
        # print(state[i])
        # print('------------------------')
        if not check_outlier_AR_based(errors):
            # exit(0)
            # print('mission_id:%d'%mission_id)
            # print(i)
            # flag = False
            return False
        X_i = np.concatenate((state[i].reshape(n,1),math.sqrt(lambd)*X_i),axis=1)
        Y_i = np.concatenate((Yk,math.sqrt(lambd)*Y_i),axis=1)
        psi_i = lambd*psi_i + np.mat(state[i].reshape(n,1))*np.mat(YkT)
        phi_i = lambd*phi_i + np.mat(Yk)*np.mat(YkT)
        # phi_i_inv = pinv(phi_i)
        # print(cond(phi_i))
        phi_i_inv = phi_i_inv / lambd - math.pow(lambd,-2)*np.mat(phi_i_inv)*np.mat(Yk)*inv(np.mat(np.identity(1)+np.mat(YkT)*np.mat(phi_i_inv)*np.mat(Yk)/lambd))*np.mat(YkT)*np.mat(phi_i_inv)
        A_i = np.mat(psi_i)*np.mat(phi_i_inv)
        i += 1
    # print(len(errors))
    # np.save("errors_50_%d"%mission_id, np.array(errors))
    # if flag == False:
    #     exit(0)
    # exit(0)
    return True


def LinearRegressionBasedLabel1(state,profile):
    state = np.array(state)
    labels = []
    i = p
    lambd = 0.99
    U = profile[0]
    X_i = state[p-1].reshape(n,1)
    Yk = computeYk(p,state,U)
    Y_i = Yk
    psi_i = np.random.random((n,n*p+q))
    phi_i = np.identity(n*p+q)
    phi_i_inv = inv(phi_i)
    A_i = np.mat(psi_i)*np.mat(phi_i_inv)
    errors = []
    
    while i < len(state):
    # while i < 14:
        Yk = computeYk(i,state,profile[i])
        YkT = np.transpose(Yk)
        X_est = np.mat(A_i)*np.mat(Yk)
        errors.append(X_est.A1-state[i])
        if not check_outlier_AR_based(errors):
            # print(i)
            # print(state[i])
            # print(X_est)
            # print('-------------------------------------------------------------------------------------------')
            return False
        X_i = np.concatenate((np.array(state[i]).reshape(n,1),math.sqrt(lambd)*X_i),axis=1)
        Y_i = np.concatenate((Yk,math.sqrt(lambd)*Y_i),axis=1)
        psi_i = lambd*psi_i + np.mat(state[i].reshape(n,1))*np.mat(YkT)
        phi_i = lambd*phi_i + np.mat(Yk)*np.mat(YkT)
        # phi_i_inv = pinv(phi_i)
        phi_i_inv = phi_i_inv / lambd - math.pow(lambd,-2)*np.mat(phi_i_inv)*np.mat(Yk)*inv(np.mat(np.identity(1)+np.mat(YkT)*np.mat(phi_i_inv)*np.mat(Yk)/lambd))*np.mat(YkT)*np.mat(phi_i_inv)
        A_i = np.mat(psi_i)*np.mat(phi_i_inv)
        i += 1
    return True        

if __name__ == '__main__':
    # dir = '/home/cedric/ArduPilot/experiment/output/PA/0/'
    # states,profiles = simulationResultClean(dir,2002,2049)
    # print(len(states))
    # np.save('/home/cedric/states',np.array(states))
    # np.save('/home/cedric/profiles',np.array(profiles))
    # with open('/home/cedric/profiles.pckl','w') as f:
        # pickle.dump(profiles,f)
    start = 0
    end = start
    states,profiles = simulationResultClean(start,end)
    # states,profiles = simulationTiny(start,end)
    labelTraces_LR(states,profiles)
    # labelTraces_LR1(states,profiles)
    exit(0)


