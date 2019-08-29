from PyDSTool import *
import random


def make_model():
    pardict_cool = {'c': 0.01, 'theta_low': 20}
    pardict_heat = {'c': 0.01, 'theta_heater': 50, 'theta_high': 30}
    vardict_cool = {'x': '-c*x', 'heating': '0'}
    vardict_heat = {'x': 'c*(theta_heater-x)', 'heating': '0'}

    cold_event_args = {'name': 'cold',
                       'eventtol': 1e-3,
                       'eventdelay': 1e-4,
                       'starttime': 0,
                       'term': True}
    cold_event = Events.makeZeroCrossEvent('x - theta_low', -1, cold_event_args, varnames=['x'], parnames=['theta_low'])
    cool_args = {'name': 'cool',
                 'pars': pardict_cool,
                 'varspecs': vardict_cool,
                 'xdomain': {'x': [0, 100], 'heating': 0.},
                 'ics': {'x': 25, 'heating': 0},
                 'tdomain': [0, Inf],
                 'algparams': {'init_step': 0.1},
                 'events': cold_event}
    cool_model = embed(Generator.Vode_ODEsystem(cool_args), name='cool', tdata=[0, 200])

    hot_event_args = {'name': 'hot',
                      'eventtol': 1e-3,
                      'eventdelay': 1e-4,
                      'starttime': 0,
                      'term': True}
    hot_event = Events.makeZeroCrossEvent('x - theta_high', 1, hot_event_args, varnames=['x'], parnames=['theta_high'])
    heat_args = {'name': 'heat',
                 'pars': pardict_heat,
                 'varspecs': vardict_heat,
                 'xdomain': {'x': [0, 100], 'heating': 1},
                 'ics': {'x': 25, 'heating': 1.},
                 'tdomain': [0, Inf],
                 'algparams': {'init_step': 0.1},
                 'events': hot_event}
    heat_model = embed(Generator.Vode_ODEsystem(heat_args), name='heat', tdata=[0, 200])

    all_model_names = ['cool', 'heat']
    cool_MI = intModelInterface(cool_model)
    heat_MI = intModelInterface(heat_model)
    cool_MI_info = makeModelInfoEntry(cool_MI, all_model_names, [('cold', 'heat')])
    heat_MI_info = makeModelInfoEntry(heat_MI, all_model_names, [('hot', 'cool')])
    model_info = makeModelInfo([cool_MI_info, heat_MI_info])

    model_args = {'name': 'thermostat', 'modelInfo': model_info}
    model = Model.HybridModel(model_args)
    return model


def physical_trace(ic_temp):
    global thermostat_model
    if ic_temp < 20:
        p = 1
        ic = {'x': ic_temp, 'heating': 1}
    elif ic_temp > 30:
        if ic_temp < 50:
            p = 2
            ic = {'x': ic_temp, 'heating': 0}
        else:
            p = 3
            ic = {'x': ic_temp, 'heating': 0}
    else:
        if random.uniform(0, 10) < 1:
            p = 4  # Faulty path
            ic = {'x': ic_temp, 'heating': 0}
        else:
            if 25 < ic_temp < 26:
                p = 5  # Faulty path
                ic = {'x': ic_temp, 'heating': 0}
            else:
                sensor_limit = random.uniform(29, 30)
                if ic_temp > sensor_limit:
                    p = 6  # Faulty path
                    ic = {'x': ic_temp, 'heating': 0}
                else:
                    p = 7
                    ic = {'x': ic_temp, 'heating': 1}
    thermostat_model.compute(trajname='Temp.', tdata=[0, 200], ics=ic, verboselevel=0)
    data = thermostat_model.sample('Temp.', dt=0.05)
    del thermostat_model['Temp.']
    return data, p


if __name__ == '__main__':
    DATA_DIR = '/Users/enyanhuang/Developer/PycharmProjects/ArdupilotDebug/data/small_exp3/'
    thermostat_model = make_model()
    for exp_i in range(2000):
        print("PA: Exp %d..." % exp_i)
        temps = []
        temp_center = random.uniform(20.5, 29.5)
        for core_i in range(4):
            t = random.uniform(temp_center-0.5, temp_center+0.5)
            temps.append(t)
            states, path = physical_trace(t)
            np.save(DATA_DIR+'PA/states_%d_%d.npy' % (exp_i, core_i), states['x'])
            with open(DATA_DIR+'PA/raw_%d_%d.txt' % (exp_i, core_i), 'a') as f:
                f.write('%d\n' % path)

    for exp_i in range(2000):
        print("AB: Exp %d..." % exp_i)
        temps = []
        temp_center = random.uniform(20.5, 29.5)
        for core_i in range(4):
            t = random.uniform(20, 30)
            temps.append(t)
            states, path = physical_trace(t)
            np.save(DATA_DIR+'AB/states_%d_%d.npy' % (exp_i, core_i), states['x'])
            with open(DATA_DIR+'AB/raw_%d_%d.txt' % (exp_i, core_i), 'a') as f:
                f.write('%d\n' % path)
