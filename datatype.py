import random
from dronekit import LocationGlobal
import copy


def randrangef(start, stop, step):
    return random.randint(0, int((stop - start) / step)) * step + start


def uniform(start, stop):
    return round(random.uniform(start, stop), 3)


class InitialProfile:

    def __init__(self, reference=None):
        if reference is None:
            self.lat = random.uniform(-36, -35)
            self.lon = random.uniform(40, 41)
            self.alt = random.uniform(400.0, 500.0)
            # self.lat = -35
            # self.lon = 40
            # self.alt = 400
            self.roll = random.uniform(-20, 20)
            self.pitch = random.uniform(-20, 20)
            self.yaw = random.uniform(10.0, 350.0)
            # self.roll = 0
            # self.pitch = 0
            # self.yaw = 0
            self.home = LocationGlobal(self.lat,self.lon,self.alt)
            self.target1 = LocationGlobal(random.uniform(self.lat+0.002, self.lat+0.003),
                                          random.uniform(self.lon+0.002, self.lon+0.003),
                                          random.uniform(self.alt+35, self.alt+40))
            self.target2 = LocationGlobal(random.uniform(self.lat+0.002, self.lat+0.003),
                                          random.uniform(self.lon-0.003, self.lon-0.002),
                                          random.uniform(self.alt-30, self.alt-20))
            self.target3 = LocationGlobal(random.uniform(self.lat+0.002, self.lat+0.003),
                                          random.uniform(self.lon+0.002, self.lon+0.003),
                                          random.uniform(self.alt+35, self.alt+40))
            self.target4 = LocationGlobal(random.uniform(self.lat+0.002, self.lat+0.003),
                                          random.uniform(self.lon - 0.003, self.lon - 0.002),
                                          random.uniform(self.alt-15, self.alt-10))
            # self.targets = [target1,target2,target3,target4]
            self.gs1 = random.randint(2, 10)
            self.gs2 = random.randint(10, 15)
            # self.params = {"WPNAV_SPEED": random.randrange(200, 900, 50),
            #                "WPNAV_RADIUS": random.randrange(110, 490, 1),
            #                "WPNAV_SPEED_UP": random.randrange(100, 500, 50),
            #                "WPNAV_SPEED_DN": random.randrange(60, 250, 10),
            #                "WPNAV_ACCEL": random.randrange(100, 250, 10),
            #                "WPNAV_ACCEL_Z": random.randrange(100, 250, 10)}
            self.params = {"WPNAV_SPEED": 500,
                           "WPNAV_RADIUS": 500,
                           "WPNAV_SPEED_UP": 200,
                           "WPNAV_SPEED_DN": 200,
                           "WPNAV_ACCEL": 300,
                           "WPNAV_ACCEL_Z": 300}

        else:
            self.lat = random.uniform(reference.lat-0.0001, reference.lat+0.0001)
            self.lon = random.uniform(reference.lon-0.0001, reference.lon+0.0001)
            self.alt = random.uniform(reference.alt-10, reference.alt+10)
            self.roll = random.uniform(reference.roll-5, reference.roll+5)
            self.pitch = random.uniform(reference.pitch-5, reference.pitch+5)
            self.yaw = random.uniform(reference.yaw-10, reference.yaw+10)
            self.target1 = reference.target1
            self.target2 = reference.target2
            self.target3 = reference.target3
            self.target4 = reference.target4
            self.gs1 = reference.gs1
            self.gs2 = reference.gs2
            self.params = copy.deepcopy(reference.params)


def generate_profiles(reference=None, cluster=True,one_profile=True):
    if one_profile:
        return [InitialProfile()]
    if cluster:
        if reference is None:
            _profiles = [InitialProfile()]
        else:
            _profiles = [reference]
        for _ in range(1, 4):
            _profiles.append(InitialProfile(reference=_profiles[0]))
        return _profiles
    else:
        return [InitialProfile(), InitialProfile(), InitialProfile(), InitialProfile()]
