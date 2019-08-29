from statsmodels.tsa.ar_model import AR
# from sklearn.metrics import mean_squared_error
import pickle
import numpy as np

def simulationResultClean(dir,index_from,index_to):
    states = []
    profiles = []
    for simulate_id in range(index_from,index_to+1):
        with open(dir + 'profiles_%d.pckl'%simulate_id) as f:
            p = pickle.load(f)[0]
            # temp_profile = []
            # for p in profile:
                # temp_profile.append([p.lat,p.lon,p.target1.lat,p.target1.lon,p.target2.lat,p.target2.lon])
            profiles.append([[p.lat,p.lon,p.alt+20],[p.target1.lat,p.target1.lon,p.target1.alt],[p.target2.lat,p.target2.lon,p.target2.alt],
            [p.target3.lon,p.target3.lat,p.target3.alt],[p.target4.lon,p.target4.lat,p.target4.alt]])
        # for trace_id in range(0,4):
            # state = np.load(dir + 'states_np_%d_%d.npy'%(simulate_id,trace_id))
            # temp.append(state)
        states.append(np.load(dir + 'states_np_%d_0.npy'%simulate_id))
        # states.append(temp)
    return states,profiles

def labelTraces(states=None,profiles=None,dir=None,start=0,end=1):
    dir = '/home/cedric/ArduPilot/experiment/output/PA/0/'
    states,profiles = simulationResultClean(dir,start,end)
    true_labels = 0
    labels = []
    for simulate_id,state in enumerate(states):
        profile = profiles[simulate_id]
        label = True
        for mission_id in range(0,len(profile)):
            state_temp = np.transpose(np.transpose(state[mission_id])[:3])
            profile_temp = profile[mission_id][:3]
            if not AR_based(state_temp,profile_temp):
                label = False
                # print(simulate_id)
                break
        if label:
            true_labels += 1
            labels.append(0)
        else:
            # print(simulate_id)
            labels.append(1)
            # break
        # break
    print('total traces:%d'%len(states))
    print('true labels:%d'%true_labels)
    return labels


def AR_based(state,profile=None):
    train,test = state[0:10],state[10:]
    model = AR(train)
    model_fit = model.fit()
    window = model_fit.k_ar
    coef = model_fit.params

    history = train[len(train)-window:]
    history = [history[i] for i in range(0,len(history))]
    predictions = []
    for t in range(len(test)):
        length = len(history)
        lag = [history[i] for i in range(length-window,length)]
        yhat = coef[0]
        for d in range(window):
            yhat += coef[d+1] * lag[window-d-1]
        obs = test[t]
        predictions.append(yhat)
        history.append(obs)
        print('predicted=%f, expected=%f'%(yhat,obs))

if __name__ == '__main__':
    labelTraces(start=2000,end=2049)