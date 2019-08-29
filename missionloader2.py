from dronekit import VehicleMode
from dronekit import LocationGlobal
import math


def get_location_metres(original_location, dNorth, dEast):
    earth_radius = 6378137.0
    dLat = dNorth/earth_radius
    dLon = dEast/(earth_radius*math.cos(math.pi*original_location.lat/180))

    newlat = original_location.lat + (dLat * 180/math.pi)
    newlon = original_location.lon + (dLon * 180/math.pi)
    return LocationGlobal(newlat, newlon,original_location.alt)


def get_distance_metres(aLocation1, aLocation2):
    dlat = aLocation2.lat - aLocation1.lat
    dlong = aLocation2.lon - aLocation1.lon
    return math.sqrt((dlat*dlat) + (dlong*dlong)) * 1.113195e5


def load_missions(sim_runner):
    def mission1():
        sim_runner.vehicle.simple_goto(sim_runner.profile.target1)

    def mission2():
        sim_runner.vehicle.mode = VehicleMode("STABILIZE")

    def mission3():
        sim_runner.vehicle.mode = VehicleMode("GUIDED")
        sim_runner.vehicle.simple_goto(sim_runner.profile.target2)

    def mission4():
        sim_runner.vehicle.mode = VehicleMode("GUIDED")
        sim_runner.vehicle.simple_goto(sim_runner.profile.target3, groundspeed=sim_runner.profile.gs1)

    def mission5():
        sim_runner.vehicle.mode = VehicleMode("ALT_HOLD")

    def mission6():
        sim_runner.vehicle.mode = VehicleMode("GUIDED")
        sim_runner.vehicle.simple_goto(sim_runner.profile.target4, groundspeed=sim_runner.profile.gs2)

    def mission7():
        sim_runner.vehicle.mode = VehicleMode("RTL")

    v = sim_runner.vehicle
    p = sim_runner.profile
    m = sim_runner.mission

    if m == 2:
        for i in range(1, 5):
            v.channels.overrides[str(i)] = 1500
    if m == 5:
        v.channels.overrides["1"] = 1400
        v.channels.overrides["2"] = 1400
        v.channels.overrides["3"] = 1500
        v.channels.overrides["4"] = 1500

    if m == 0 and v.location.global_relative_frame.alt > 9:
        mission1()
        sim_runner.mission += 1
    if m == 1 and get_distance_metres(v.location.global_frame, p.target1) < 1:
        mission2()
        sim_runner.mission += 1
    if m == 2 and abs(v.location.global_frame.alt - p.target1.alt) > 5:
        v.channels.overrides = {}
        mission3()
        sim_runner.mission += 1
    if m == 3 and get_distance_metres(v.location.global_frame, p.target2) < 1:
        mission4()
        sim_runner.mission += 1
    if m == 4 and get_distance_metres(v.location.global_frame, p.target3) < 1:
        mission5()
        sim_runner.mission += 1
    if m == 5 and get_distance_metres(v.location.global_frame, p.target3) > 5:
        v.channels.overrides = {}
        mission6()
        sim_runner.mission += 1
    if m == 6 and get_distance_metres(v.location.global_frame, p.target4) < 1:
        mission7()
        sim_runner.mission += 1

