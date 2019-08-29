from dronekit import VehicleMode, LocationGlobal
import math


def get_location_metres(original_location, dNorth, dEast):
    earth_radius = 6378137.0
    dLat = dNorth/earth_radius
    dLon = dEast/(earth_radius*math.cos(math.pi*original_location.lat/180))

    newlat = original_location.lat + (dLat * 180/math.pi)
    newlon = original_location.lon + (dLon * 180/math.pi)
    return LocationGlobal(newlat, newlon,original_location.alt)


def get_distance_metres(aLocation1, aLocation2): ### Cedric: it will not be arrurate over large distance and close to the earth's poles.
    dlat = aLocation2.lat - aLocation1.lat
    dlong = aLocation2.lon - aLocation1.lon
    return math.sqrt((dlat*dlat) + (dlong*dlong)) * 1.113195e5

class Mission(object):
    def __init__(self):
        self.first_enter = True

    def run(self):
        pass

    def is_completed(self):
        return True

class Mission1(Mission):
    def __init__(self, ):
        super(Mission1,self).__init__()

    def run(self,sim_runner):
        if self.first_enter:
            sim_runner.vehicle.mode = VehicleMode('GUIDED')
            sim_runner.vehicle.simple_takeoff(20)
            self.first_enter = False

    def is_completed(self,sim_runner):
        if sim_runner.current_time >= 10:
            return True
        else:
            return False

class Mission2(Mission):
    def __init__(self, ):
        super(Mission2,self).__init__()

    def run(self,sim_runner):
        if self.first_enter:
            self.first_enter = False            
            sim_runner.vehicle.channels.overrides['1'] = 1400
            sim_runner.vehicle.channels.overrides['2'] = 1400
            sim_runner.vehicle.channels.overrides['3'] = 1500
            sim_runner.vehicle.channels.overrides['4'] = 1500
            sim_runner.vehicle.mode = VehicleMode('LOITER')
        ## overrides throttle everytime to hold the altitude
        ### Mode like Loiter , Alt_Hode , Stabilize also need to override throttle everytime
        sim_runner.vehicle.channels.overrides['3'] = 1500

    def is_completed(self,sim_runner):
        if sim_runner.current_time >= 15:
            return True
        else:
            return False

class Mission3(Mission):
    def __init__(self, ):
        super(Mission3,self).__init__()

    def run(self,sim_runner):
        if self.first_enter:
            self.first_enter = False
            sim_runner.vehicle.channels.overrides = {}
            sim_runner.vehicle.channels.overrides['3'] = 1500
            sim_runner.vehicle.mode = VehicleMode('STABILIZE')

        ## overrides throttle everytime to hold the altitude
        sim_runner.vehicle.channels.overrides['3'] = 1500

    def is_completed(self,sim_runner):
        if sim_runner.current_time >= 20:
            return True
        else:
            return False

class Mission4(Mission):
    def __init__(self, ):
        super(Mission4,self).__init__()

    def run(self,sim_runner):
        if self.first_enter:
            self.first_enter = False
            sim_runner.vehicle.mode = VehicleMode('GUIDED')
            sim_runner.vehicle.simple_goto(sim_runner.profile.target1)

    def is_completed(self,sim_runner):
        if sim_runner.current_time >= 40:
            return True
        else:
            return False

class Mission5(Mission):
    def __init__(self, ):
        super(Mission5,self).__init__()

    def run(self,sim_runner):
        if self.first_enter:
            self.first_enter = False            
            sim_runner.vehicle.channels.overrides['1'] = 1400
            sim_runner.vehicle.channels.overrides['2'] = 1400
            sim_runner.vehicle.channels.overrides['3'] = 1500
            sim_runner.vehicle.channels.overrides['4'] = 1500
            sim_runner.vehicle.mode = VehicleMode('ALT_HOLD')
        ## overrides throttle everytime to hold the altitude
        sim_runner.vehicle.channels.overrides['3'] = 1500

    def is_completed(self,sim_runner):
        if sim_runner.current_time >= 45:
            return True
        else:
            return False

class Mission6(Mission):
    def __init__(self,):
        super(Mission6,self).__init__()

    def run(self,sim_runner):
        if self.first_enter:
            self.first_enter = False
            sim_runner.vehicle.mode = VehicleMode('GUIDED')
            sim_runner.vehicle.simple_goto(sim_runner.profile.target2)
    
    def is_completed(self,sim_runner):
        if sim_runner.current_time >= 65:
            return True
        else:
            return False

class Mission7(Mission):
    def __init__(self,):
        super(Mission7,self).__init__()

    def run(self,sim_runner):
        if self.first_enter:
            self.first_enter = False
            sim_runner.vehicle.mode = VehicleMode('RTL')

    def is_completed(self,sim_runner):
        if sim_runner.current_time >= 100:
            return True
        else:
            return False