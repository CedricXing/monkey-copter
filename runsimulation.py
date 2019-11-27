from dronekit_sitl import SITL
from dronekit import connect, APIException, VehicleMode, LocationGlobalRelative, LocationGlobal, Command
import time
import numpy as np
import pickle
from datatype import *
# from mission import *
from mission1 import *
import sys
from getopt import getopt, GetoptError
from pymavlink import mavutil
from ConfigParser import ConfigParser

bug_id = 0
    
def print_info(vehicle):
    print('current altitude:%s'%str(vehicle.location.global_relative_frame.alt))
    print('current lat:%s'%vehicle.location.global_relative_frame.lat)
    print('current lon:%s'%vehicle.location.global_relative_frame.lon)
    print('vehicle attitude:%s'%vehicle.attitude)
    print('vehicle velocity:%s'%vehicle.velocity)
    print("mode:%s"%vehicle.mode.name)

class SimRunner:
    def arm_vehicle(self):
        arm_time = 0
        self.vehicle.armed = True

        while not self.vehicle.armed:
            time.sleep(0.2)
            print('waiting for armed...')
            arm_time += 0.2
            if arm_time > 30:
                print("Arm fail")
                self.vehicle.close()
                self.sitl.stop()
                self.sitl.p.stdout.close()
                self.sitl.p.stderr.close()
                self.ready = False
                return

    def __init__(self, sample_id, core_id, initial_profile,config):
        ardupilot_dir = config['root_dir'] + 'experiment/elf/'
        out_dir = config['root_dir'] + 'experiment/output/'
        self.elf_dir = "%s%d/" % (ardupilot_dir, 0)
        self.exp_out_dir = "%s%s/%d/" % (out_dir, 'PA', 0)
        self.ready = True
        self.states = []
        self.profiles = []
        self.sim_id = "%d_%d" % (sample_id, core_id)
        copter_file = self.elf_dir + "ArduCopter.elf"
        self.delta = 0.1
        self.sitl = SITL(path=copter_file)
        home_str = "%.6f,%.6f,%.2f,%d" % (initial_profile.lat, initial_profile.lon, initial_profile.alt,
                                          initial_profile.yaw)
        sitl_args = ['-S', '-I%d' % bug_id, '--home='+home_str, '--model',
                     '+', '--speedup=1', '--defaults='+self.elf_dir+'copter.parm']
        self.sitl.launch(sitl_args, await_ready=True, restart=True, wd=self.elf_dir)
        port_number = 5760 + bug_id * 10
        self.missions = [Mission1(),Mission2(),Mission3(),Mission4(),Mission5()]
        try:
            self.vehicle = connect('tcp:127.0.0.1:%d' % port_number, wait_ready=True,rate=10)
        except APIException:
            print("Cannot connect")
            self.sitl.stop()
            self.sitl.p.stdout.close()
            self.sitl.p.stderr.close()
            self.ready = False
            return
        for k, v in initial_profile.params.items():
            self.vehicle.parameters[k] = v
        while not self.vehicle.is_armable:
            print('initializing...')
            time.sleep(1)
        self.arm_vehicle()
        time.sleep(2)
        print(self.vehicle.version)

    def run(self):
        takeoff_mission = self.missions[0]
        takeoff_mission.run(self)
        time.sleep(10)
        temp_state = []
        waypoint_num = 3
        T = 5

        ### 4 missions. Each mission has 3 waypoints. For each waypoint, the vehicle runs for 2 seconds 
        for i in range(0,4):
            current_location = self.vehicle.location.global_frame
            if i % 2 == 0:
                target_delta = [random.uniform(0.0002,0.0003)/waypoint_num,random.uniform(0.0002,0.0003)/waypoint_num,random.uniform(20,30)/waypoint_num]
            else:
                target_delta = [random.uniform(0.0002,0.0003)/waypoint_num,random.uniform(-0.0003,-0.0002)/waypoint_num,random.uniform(-30,-20)/waypoint_num]
            # count = 0
            for j in range(1,waypoint_num+1):
                profile = LocationGlobal(current_location.lat+target_delta[0]*j,current_location.lon+target_delta[1]*j,current_location.alt+target_delta[2]*j)
                self.profiles.append([profile.lat,profile.lon,profile.alt])
                self.vehicle.simple_goto(profile)
                current_t = 0
                temp_state = []
                
                while current_t < T:
                    ### record the [lat,lon,alt,pitch,yaw,roll,velocity] every 0.1 seconds
                    temp_state.append([self.vehicle.location.global_frame.lat,self.vehicle.location.global_frame.lon,self.vehicle.location.global_frame.alt
                    ,self.vehicle.attitude.pitch,self.vehicle.attitude.yaw,self.vehicle.attitude.roll,self.vehicle.velocity[0],self.vehicle.velocity[1],self.vehicle.velocity[2]])
                    time.sleep(0.1)
                    current_t += 0.1
                    # count += 1
                    # print(count)
                    # print_info(self.vehicle)
                self.states.append(temp_state)

        self.vehicle.close()
        self.sitl.stop()

        np.save(self.exp_out_dir + "states_np_%s" % self.sim_id, np.array(self.states))
        np.save(self.exp_out_dir + "profiles_np_%s" % self.sim_id, np.array(self.profiles))

        print("Output Execution Path...")
        with open(self.exp_out_dir + "raw_%s.txt" % self.sim_id, "w") as execution_file:
            ep_line = self.sitl.stdout.readline(0.01)
            while ep_line is not None:
                execution_file.write(ep_line)
                ep_line = self.sitl.stdout.readline(0.01)

        self.sitl.p.stdout.close()
        self.sitl.p.stderr.close()
        print("Simulation %s completed." % self.sim_id)

    def run1(self):
        takeoff_mission = self.missions[0]
        takeoff_mission.run(self)
        time.sleep(10)
        home_location = self.vehicle.location.global_frame
        temp_state = []
        waypoint_num = 3
        T = 5
        ## first mission : guided mode
        for i in range(0,5):
            current_location = self.vehicle.location.global_frame
            if i % 2 == 0:
                target_delta = [random.uniform(0.0002,0.0003)/waypoint_num,random.uniform(0.0002,0.0003)/waypoint_num,random.uniform(20,30)/waypoint_num]
            else:
                target_delta = [random.uniform(0.0002,0.0003)/waypoint_num,random.uniform(-0.0003,-0.0002)/waypoint_num,random.uniform(-30,-20)/waypoint_num]
            for j in range(1,waypoint_num+1):
                profile = LocationGlobal(current_location.lat+target_delta[0]*j,current_location.lon+target_delta[1]*j,current_location.alt+target_delta[2]*j)
                self.profiles.append([profile.lat,profile.lon,profile.alt])
                self.vehicle.simple_goto(profile)
                current_t = 0
                temp_state = []
                while current_t < T:
                    temp_state.append([self.vehicle.location.global_frame.lat,self.vehicle.location.global_frame.lon,self.vehicle.location.global_frame.alt
                    ,self.vehicle.attitude.pitch,self.vehicle.attitude.yaw,self.vehicle.attitude.roll,self.vehicle.velocity[0],self.vehicle.velocity[1],self.vehicle.velocity[2]])
                    time.sleep(0.1)
                    current_t += 0.1
                self.states.append(temp_state)
        
        ## second mission : acro mode
        self.vehicle.channels.overrides['1'] = 1400
        self.vehicle.channels.overrides['2'] = 1400
        self.vehicle.channels.overrides['3'] = 1500
        self.vehicle.channels.overrides['4'] = 1500
        self.vehicle.mode = VehicleMode('ACRO')
        for i in range(0,waypoint_num):
            current_location = self.vehicle.location.global_frame
            self.profiles.append([current_location.lat,current_location.lon,current_location.alt])
            current_t = 0
            temp_state = []
            while current_t < T:
                self.vehicle.channels.overrides['3'] = 1500
                temp_state.append([self.vehicle.location.global_frame.lat,self.vehicle.location.global_frame.lon,self.vehicle.location.global_frame.alt
                    ,self.vehicle.attitude.pitch,self.vehicle.attitude.yaw,self.vehicle.attitude.roll,self.vehicle.velocity[0],self.vehicle.velocity[1],self.vehicle.velocity[2]])
                time.sleep(0.1)
                current_t += 0.1
                # print_info(self.vehicle)
            self.states.append(temp_state)
        
        ## third mission : auto mode
        cmds = self.vehicle.commands
        cmds.clear()
        waypoint_in_auto = [random.uniform(0.0002,0.0003)/waypoint_num,random.uniform(0.0002,0.0003)/waypoint_num,random.uniform(20,30)/waypoint_num]
        for j in range(1,waypoint_num+1):
            profile = LocationGlobal(current_location.lat+target_delta[0]*j,current_location.lon+target_delta[1]*j,current_location.alt+target_delta[2]*j)
            self.profiles.append([profile.lat,profile.lon,profile.alt])
            cmds.add(Command(0,0,0,mavutil.mavlink.MAV_FRAME_GLOBAL_RELATIVE_ALT,mavutil.mavlink.MAV_CMD_NAV_WAYPOINT,0,0,0,0,0,0,profile.lat,profile.lon,profile.alt))
        cmds.upload()
        self.vehicle.commands.next = 0
        self.vehicle.mode = VehicleMode('AUTO')
        for j in range(0,waypoint_num):
            current_t = 0
            temp_state = []
            while current_t < T:
                temp_state.append([self.vehicle.location.global_frame.lat,self.vehicle.location.global_frame.lon,self.vehicle.location.global_frame.alt
                ,self.vehicle.attitude.pitch,self.vehicle.attitude.yaw,self.vehicle.attitude.roll,self.vehicle.velocity[0],self.vehicle.velocity[1],self.vehicle.velocity[2]])
                time.sleep(0.1)
                current_t += 0.1
            self.vehicle.commands.next += 1
            self.states.append(temp_state)

        #### fourth mission : rtl mode
        self.vehicle.mode = VehicleMode('RTL')
        for j in range(0,waypoint_num):
            self.profiles.append([home_location.lat,home_location.lon,home_location.alt])
            current_t = 0
            temp_state = []
            while current_t < T:
                temp_state.append([self.vehicle.location.global_frame.lat,self.vehicle.location.global_frame.lon,self.vehicle.location.global_frame.alt
                ,self.vehicle.attitude.pitch,self.vehicle.attitude.yaw,self.vehicle.attitude.roll,self.vehicle.velocity[0],self.vehicle.velocity[1],self.vehicle.velocity[2]])
                time.sleep(0.1)
                current_t += 0.1
            self.states.append(temp_state)
        self.vehicle.close()
        self.sitl.stop()

        np.save(self.exp_out_dir + "states_np_%s" % self.sim_id, np.array(self.states))
        np.save(self.exp_out_dir + "profiles_np_%s" % self.sim_id, np.array(self.profiles))

        print("Output Execution Path...")
        with open(self.exp_out_dir + "raw_%s.txt" % self.sim_id, "w") as execution_file:
            ep_line = self.sitl.stdout.readline(0.01)
            while ep_line is not None:
                execution_file.write(ep_line)
                ep_line = self.sitl.stdout.readline(0.01)

        self.sitl.p.stdout.close()
        self.sitl.p.stderr.close()
        print("Simulation %s completed." % self.sim_id)

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

def run_sim(config,mission_no):
    sim_start = config['start']
    sim_end = config['end']
    
    ### start simulations, (sim_end - sim_start) rounds for all
    while sim_start < sim_end:
        print("simulation round %d-----------------------------------------\n" %sim_start)
        try:
            ### generate only one profile as the home location in default
            profiles_generated = generate_profiles()
            for core_cnt, profile in enumerate(profiles_generated):
                sim = SimRunner(sim_start, core_cnt, profile,config)
                if sim.ready:
                    if mission_no == 0: ### run four basic guided missions
                        sim.run()
                    else: ### this is designed for bug subjects that contain bugs in rtl, auto, arco...
                        sim.run1()
                else:
                    print("Not ready for mission")
                    break
            else:
                sim_start += 1
        except IOError:
            print('io error')
            continue

if __name__ == "__main__":
    config = parserConfig()
    run_sim(config,0)
