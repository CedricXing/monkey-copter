# Randomly insert some bugs in two libaray files (AC_AttitudeControl/AC_PosControl.cpp and AC_WPNav/AC_WPNav.cpp)

import os
import random

ARDUPILOT_DIR = "/home/cedric/Desktop/arduPilot/"

bug_num_to_insert = 3

# target_statements = [
#     '_pos_target.x += _vel_desired.x * nav_dt;', # 0
#     '_pos_target.y += _vel_desired.y * nav_dt;', # 1
#     '_vehicle_horiz_vel.x = _inav.get_velocity().x;',# 2
#     '_vehicle_horiz_vel.y = _inav.get_velocity().y;', # 3
#     '_accel_desired.z = (_vel_target.z - _vel_last.z) / _dt;', # 4
#     '_vel_error.z = _vel_error_filter.apply(_vel_target.z - curr_vel.z, _dt);', # 5
#     '_accel_error.z = _accel_target.z - z_accel_meas;', # 6
#     '_pos_error.x = _pos_target.x - curr_pos.x;', # 7
#     '_pos_error.y = _pos_target.y - curr_pos.y;' # 8
#     # "_vel_target.x = _speed_cms * _vel_target.x/vel_total;",	# 0
#     # "_vel_target.y = _speed_cms * _vel_target.y/vel_total;",	# 1
#     # "leash_length = POSCONTROL_LEASH_LENGTH_MIN;",				# 2
#     # "_pos_error.z = -_leash_down_z;",							# 3
#     # "_pos_target.z = curr_alt + _leash_up_z;",					# 4
#     # "_vel_target.z = _speed_down_cms;",							# 7
#     # "stopping_point.z = curr_pos_z + (linear_distance + curr_vel_z*curr_vel_z/(2.0f*_accel_z_cms));",	# 8
#     # "stopping_point.z = curr_pos_z - (linear_distance + curr_vel_z*curr_vel_z/(2.0f*_accel_z_cms));",	# 9
#     # "track_desired_max = track_covered;",						# 10
#     # "_track_accel = 0;",										# 11
#     # "_track_speed = 0;",										# 12
#     # "_pos_delta_unit.x = 0;",									# 13
#     # "_slow_down_dist = 0.0f;"									# 14
# ]

# mutated_statements = [
#     '/* BUG!! */ _pos_target.x = _vel_desired.x * nav_dt;',	# 0
#     '/* BUG!! */ _pos_target.y = _vel_desired.y * nav_dt;', # 1
#     '/* BUG!! */ _vehicle_horiz_vel.x = _inav.get_velocity().y;', # 2
#     '/* BUG!! */ _vehicle_horiz_vel.y = _inav.get_velocity().x;', # 3
#     '/* BUG!! */ _accel_desired.z = (_vel_target.z - _vel_last.z);', # 4
#     '/* BUG!! */ _vel_error.z = _vel_error_filter.apply(curr_vel.z - _vel_target.z, _dt);', # 5
#     '/* BUG!! */ //_accel_error.z = _accel_target.z - z_accel_meas;', # 6
#     '/* BUG!! */ _pos_error.x = curr_pos.x - _pos_target.x;',# 7
#     '/* BUG!! */ _pos_error.y = curr_pos.y - _pos_target.y;'# 8
    
#     # "/* BUG!! */ _vel_target.x = _speed_cms / _vel_target.x/vel_total;",	# 0
#     # "/* BUG!! */ _vel_target.y = _speed_cms / _vel_target.y/vel_total;",	# 1
#     # "/* BUG!! */ // leash_length = POSCONTROL_LEASH_LENGTH_MIN;",			# 2
#     # "/* BUG!! */ _pos_error.z = _leash_down_z;",							# 3
#     # "/* BUG!! */ _pos_target.z = curr_alt - _leash_up_z;",					# 4
#     # "/* BUG!! */ _vel_target.z = _speed_up_cms;",							# 7
#     # "/* BUG!! */ stopping_point.z = curr_pos_z - (linear_distance + curr_vel_z*curr_vel_z/(2.0f*_accel_z_cms));",	# 8
#     # "/* BUG!! */ stopping_point.z = curr_pos_z + (linear_distance + curr_vel_z*curr_vel_z/(2.0f*_accel_z_cms));",	# 9
#     # "/* BUG!! */ // track_desired_max = track_covered;",					# 10
#     # "/* BUG!! */ // _track_accel = 0;",										# 11
#     # "/* BUG!! */ // _track_speed = 0;",										# 12
#     # "/* BUG!! */ return false;",											# 13
#     # "/* BUG!! */ // _slow_down_dist = 0.0f;"								# 14
# ]
target_statements = [
    'linear_distance = _accel_z_cms / (2.0f * _p_pos_z.kP() * _p_pos_z.kP());', # 0 line:430
    'leash_length = POSCONTROL_LEASH_LENGTH_MIN', # 1 good line:1118
    'stopping_point.z = curr_pos_z + (linear_distance + curr_vel_z * curr_vel_z / (2.0f * _accel_z_cms));', # 2 not good 427
    'stopping_point.z = curr_pos_z - (linear_distance + curr_vel_z * curr_vel_z / (2.0f * _accel_z_cms))', # 3 not good 432
    '_pos_target.z = curr_alt + _leash_up_z;', # 4 514 good
    '_pos_error.x = _pos_target.x - curr_pos.x;', # 5 981 high frequency
    '_pos_error.y = _pos_target.y - curr_pos.y;', # 6 981 not good
    '_accel_cms = accel_cmss;', # 7 611
    '_vel_target.z = _speed_down_cms;', # 8 532
    '_vel_target.z = _speed_up_cms;', # 9 536
    'vector_x *= (max_length / vector_length);', #10 # 1167
    'vector_y *= (max_length / vector_length);', #11 #1167
    '_speed_cms = speed_cms;', #12 621
    '', #13
    '', #14
    '', #15
    '', #16
    '', #17
    '', #18
    '', #19
    '', #20
    # nav
    '_track_accel = 0;', # 21 545
    '_track_speed = 0;',# 22 545
    '_pos_delta_unit.x = 0;', # 23  237
    '_slow_down_dist = 0.0f;', # 24 1004
    'Vector3f dist_to_dest = (curr_pos - Vector3f(0,0,terr_offset)) - _destination;', # 25 442
    '_track_leash_length = leash_z/pos_delta_unit_z;', # 26  553
    '_limited_speed_xy_cms = MIN(_limited_speed_xy_cms,', # 27 399
    '_limited_speed_xy_cms -= 2.0f * _track_accel * dt;', # 28  414
    '_limited_speed_xy_cms = 0;', # 29 381
    '_track_leash_length = WPNAV_LEASH_LENGTH_MIN;', # 30 544
    '_track_leash_length = _pos_control.get_leash_xy()/pos_delta_unit_xy;',# 31 545
    '_track_leash_length = leash_z/pos_delta_unit_z;',# 32 549
    '_track_leash_length = MIN(leash_z/pos_delta_unit_z', # 33 553



    # 'return Vector3f(error.x * p, error.y * p, error.z);', # 1 low frequency
    # '_vehicle_horiz_vel.x = _inav.get_velocity().x;',# 2
    # 'vel_xy_i = _pid_vel_xy.get_i_shrink();', # 3 high frequency
    # '_speed_cms = speed_cms;', # 5 not good 
    # '_vel_error.z = _vel_error_filter.apply(_vel_target.z - curr_vel.z, _dt);', # 5 serious bug
    
    
      
    # 'linear_distance = _accel_cms / (2.0f * kP * kP);', # 8 
    # '_pos_target.x = curr_pos.x + _pos_error.x;', # 9
    # "_vel_target.x = _speed_cms * _vel_target.x/vel_total;",	# 0
    # "_vel_target.y = _speed_cms * _vel_target.y/vel_total;",	# 1
    # "leash_length = POSCONTROL_LEASH_LENGTH_MIN;",				# 2
    # "_pos_error.z = -_leash_down_z;",							# 3
    # "_pos_target.z = curr_alt + _leash_up_z;",					# 4
    # "_vel_target.z = _speed_down_cms;",							# 7
    # "stopping_point.z = curr_pos_z + (linear_distance + curr_vel_z*curr_vel_z/(2.0f*_accel_z_cms));",	# 8
    # "stopping_point.z = curr_pos_z - (linear_distance + curr_vel_z*curr_vel_z/(2.0f*_accel_z_cms));",	# 9
    # "track_desired_max = track_covered;",						# 10
    # "_track_accel = 0;",										# 11
    # "_track_speed = 0;",										# 12
    # "_pos_delta_unit.x = 0;",									# 13
    # "_slow_down_dist = 0.0f;"									# 14
]

mutated_statements = [
    '/* BUG!! */ linear_distance = _accel_z_cms / (2.0f * _p_pos_z.kP());', # 0
    '/* BUG!! */ //leash_length = POSCONTROL_LEASH_LENGTH_MIN', # 1
    '/* BUG!! */ stopping_point.z =curr_pos_z - (linear_distance + curr_vel_z * curr_vel_z / (2.0f * _accel_z_cms));', # 2
    '/* BUG!! */ stopping_point.z = curr_pos_z + (linear_distance + curr_vel_z * curr_vel_z / (2.0f * _accel_z_cms));', # 3
    '/* BUG!! */ _pos_target.z = curr_alt - _leash_up_z;', # 4
    '/* BUG!! */ _pos_error.x = curr_pos.x - _pos_target.x;', # 5
    '/* BUG!! */ _pos_error.y = curr_pos.y - _pos_target.y;', # 6
    '/* BUG!! */ _accel_cms = -accel_cmss;', # 7
    '/* BUG!! */ _vel_target.z =_speed_up_cms;', # 8
    '/* BUG!! */ _vel_target.z = _speed_down_cms;', # 9
    '/* BUG!! */ vector_x = (max_length / vector_length);', # 10
    '/* BUG!! */ vector_y = (max_length / vector_length);', # 11
    '/* BUG!! */ _speed_cms = -speed_cms;', #12
    '', #13
    '', #14
    '', #15
    '', #16
    '', #17
    '', #18
    '', #19
    '', #20
    #nav
    '/* BUG!! */ //_track_accel = 0;', # 21
    '/* BUG!! */ //_track_speed = 0;', # 22
    '/* BUG!! */ return false;', # 23
    '/* BUG!! */ //_slow_down_dist = 0.0f;', # 24
    '/* BUG!! */ Vector3f dist_to_dest = (curr_pos + Vector3f(0,0,terr_offset)) - _destination;', # 25
    '/* BUG!! */ //_track_leash_length = leash_z/pos_delta_unit_z;', # 26
    '/* BUG!! */ _limited_speed_xy_cms = MAX(_limited_speed_xy_cms,', # 27
    '/* BUG!! */ _limited_speed_xy_cms += 2.0f * _track_accel * dt;', # 28
    '/* BUG!! */ //_limited_speed_xy_cms = 0;', # 29 380
    '/* BUG!! */ //_track_leash_length = WPNAV_LEASH_LENGTH_MIN;', # 30 545
    '/* BUG!! */ //_track_leash_length = _pos_control.get_leash_xy()/pos_delta_unit_xy;',# 31 549
    '/* BUG!! */ //_track_leash_length = leash_z/pos_delta_unit_z;',# 32 553
    '/* BUG!! */ //_track_leash_length = MIN(leash_z/pos_delta_unit_z', # 33 557





    # '/* BUG!! */ return Vector3f(error.x * p, error.y, error.z);',	# 1
    # '/* BUG!! */ _vehicle_horiz_vel.x = _inav.get_velocity().y;', # 2
    # '/* BUG!! */ //vel_xy_i = _pid_vel_xy.get_i_shrink();', # 3
    
    
    # '/* BUG!! */ //_speed_cms = speed_cms;', # 5
    # '/* BUG!! */ _vel_error.z = _vel_error_filter.apply(curr_vel.z - _vel_target.z, _dt);', # 5
    
    # 7
    # '/* BUG!! */ linear_distance = _accel_cms / (kP * kP);',# 8
    # '/* BUG!! */ _pos_target.x = curr_pos.x - _pos_error.x;',# 9
    
    
    
    # "/* BUG!! */ _vel_target.x = _speed_cms / _vel_target.x/vel_total;",	# 0
    # "/* BUG!! */ _vel_target.y = _speed_cms / _vel_target.y/vel_total;",	# 1
    # "/* BUG!! */ // leash_length = POSCONTROL_LEASH_LENGTH_MIN;",			# 2
    # "/* BUG!! */ _pos_error.z = _leash_down_z;",							# 3
    # "/* BUG!! */ _pos_target.z = curr_alt - _leash_up_z;",					# 4
    # "/* BUG!! */ _vel_target.z = _speed_up_cms;",							# 7
    # "/* BUG!! */ stopping_point.z = curr_pos_z - (linear_distance + curr_vel_z*curr_vel_z/(2.0f*_accel_z_cms));",	# 8
    # "/* BUG!! */ stopping_point.z = curr_pos_z + (linear_distance + curr_vel_z*curr_vel_z/(2.0f*_accel_z_cms));",	# 9
    # "/* BUG!! */ // track_desired_max = track_covered;",					# 10
    # "/* BUG!! */ // _track_accel = 0;",										# 11
    # "/* BUG!! */ // _track_speed = 0;",										# 12
    # "/* BUG!! */ return false;",											# 13
    # "/* BUG!! */ // _slow_down_dist = 0.0f;"								# 14
]
bug_group = [
    {"file": "libraries/AC_AttitudeControl/AC_PosControl.cpp",
     "location": "libraries/AC_AttitudeControl/AC_PosControl.cpp-pos_to_rate_xy-887",
     "indices": [0],
     "start" : 100000,
     "lineno" : [430],
     "type": 'Semantic'}, #0
     {"file": "libraries/AC_AttitudeControl/AC_PosControl.cpp",
     "location": "libraries/AC_AttitudeControl/AC_PosControl.cpp-pos_to_rate_xy-887",
     "indices": [1],
     "start" : 200000,
     "lineno" : [1118],
     "type": 'Semantic'}, #1
    {"file": "libraries/AC_AttitudeControl/AC_PosControl.cpp",
     "location": "libraries/AC_AttitudeControl/AC_PosControl.cpp-calc_leash_length-1048",
     "indices": [2,3],
     "start" : 300000,
     "lineno" : [427,432],
     "type": 'Forget to assign'}, #2
    {"file": "libraries/AC_AttitudeControl/AC_PosControl.cpp",
     "location": "libraries/AC_AttitudeControl/AC_PosControl.cpp-pos_to_rate_z-374",
     "indices": [4],
     "start" : 400000,
     "lineno" : [514],
     "type": 'Semantic'}, #3
    {"file": "libraries/AC_AttitudeControl/AC_PosControl.cpp",
     "location": "libraries/AC_AttitudeControl/AC_PosControl.cpp-pos_to_rate_z-369",
     "indices": [5,6],
     "start" : 500000,
     "lineno" : [981],
     "type": 'Semantic'}, #4
    {"file": "libraries/AC_AttitudeControl/AC_PosControl.cpp",
     "location": "libraries/AC_AttitudeControl/AC_PosControl.cpp-pos_to_rate_xy-833",
     "indices": [7],
     "start" : 600000,
     "lineno" : [611],
     'type': 'Semantic'}, #5
    # {"file": "libraries/AC_AttitudeControl/AC_PosControl.cpp",
    #  "location": "libraries/AC_AttitudeControl/AC_PosControl.cpp-pos_to_rate_z-387",
    #  "indices": [8,9],
    #  "start" : 700000,
    #  "lineno" : [532,536],
    #  "type": 'Semantic'}, #6
    {"file": "libraries/AC_AttitudeControl/AC_PosControl.cpp",
     "location": "libraries/AC_AttitudeControl/AC_PosControl.cpp-get_stopping_point_z-290",
     "indices": [10,11],
     "start" : 800000,
     "lineno" : [1167],
     "type": 'Semantic'}, #6
    # {"file": "libraries/AC_AttitudeControl/AC_PosControl.cpp",
    #  "location": "libraries/AC_AttitudeControl/AC_PosControl.cpp-get_stopping_point_z-292",
    #  "indices": [12],
    #  "start" : 900000,
    #  "lineno" : [621],
    #  "type": 'Semantic'}, #8





    {"file": "libraries/AC_WPNav/AC_WPNav.cpp",
     "location": "libraries/AC_WPNav/AC_WPNav.cpp-advance_wp_target_along_track-XXX",
     "indices": [21,22],
     "start" : 2000000,
     "lineno" : [545],
     "type": "Forget to assign"}, #7
    {"file": "libraries/AC_WPNav/AC_WPNav.cpp",
     "location": "libraries/AC_WPNav/AC_WPNav.cpp-calculate_wp_leash_length-814",
     "indices": [23],
     "start" : 2100000,
     "lineno" : [237],
     "type": 'Uninitialized'}, #8
    {"file": "libraries/AC_WPNav/AC_WPNav.cpp",
     "location": "libraries/AC_WPNav/AC_WPNav.cpp-set_wp_origin_and_destination-495",
     "indices": [24],
     "start" : 2200000,
     "lineno" : [1004],
     "type": "Forget to assign"}, #9
    {"file": "libraries/AC_WPNav/AC_WPNav.cpp",
     "location": "libraries/AC_WPNav/AC_WPNav.cpp-calc_slow_down_distance-1285",
     "indices": [25],
     "start" : 2300000,
     "lineno" : [442],
     "type": 'Uninitialized'}, #10
     {"file": "libraries/AC_WPNav/AC_WPNav.cpp",
     "location": "libraries/AC_WPNav/AC_WPNav.cpp-calc_slow_down_distance-1285",
     "indices": [26],
     "start" : 2400000,
     "lineno" : [553],
     "type": 'Uninitialized'}, #11
     {"file": "libraries/AC_WPNav/AC_WPNav.cpp",
     "location": "libraries/AC_WPNav/AC_WPNav.cpp-calc_slow_down_distance-1285",
     "indices": [27],
     "start" : 2500000,
     "lineno" : [399],
     "type": 'Uninitialized'}, #12
     {"file": "libraries/AC_WPNav/AC_WPNav.cpp",
     "location": "libraries/AC_WPNav/AC_WPNav.cpp-calc_slow_down_distance-1285",
     "indices": [28],
     "start" : 2600000,
     "lineno" : [414],
     "type": 'Uninitialized'}, #13
     {"file": "libraries/AC_WPNav/AC_WPNav.cpp",
     "location": "libraries/AC_WPNav/AC_WPNav.cpp-calc_slow_down_distance-1285",
     "indices": [29],
     "start" : 2700000,
     "lineno" : [381],
     "type": 'Uninitialized'}, #14
     {"file": "libraries/AC_WPNav/AC_WPNav.cpp",
     "location": "libraries/AC_WPNav/AC_WPNav.cpp-calc_slow_down_distance-1285",
     "indices": [30,31,32],
     "start" : 2800000,
     "lineno" : [545,549,553],#,557],
     "type": 'Uninitialized'}, #15
    {"file": "libraries/AC_WPNav/AC_WPNav.cpp",
     "location": "libraries/AC_WPNav/AC_WPNav.cpp-calc_slow_down_distance-1285",
     "indices": [],
     "start" : 3000000,
     "lineno" : [],
     "type": 'no bug'}, #16
]

real_life_target_bugs = [
    'float acro_level_mix = constrain_float(1-float(MAX(MAX(abs(roll_in), abs(pitch_in)), abs(yaw_in))/4500.0), 0, 1)*ahrs.cos_pitch();', # 0 mode acro
    '/* 0 */ loiter_nav->clear_pilot_desired_acceleration();',# 1 mode auto
    '/* 1 */ loiter_nav->clear_pilot_desired_acceleration();',# 2 mode auto
    'loiter_nav->clear_pilot_desired_acceleration();',# 3 mode rtl
    'if (motors->get_spool_state() > AP_Motors::SpoolState::GROUND_IDLE)', # 4
    '_rate_target_ang_vel.x += constrain_float(attitude_error_vector.y, -M_PI / 4, M_PI / 4) * _ahrs.get_gyro().z;', # 5
    '_rate_target_ang_vel.y += -constrain_float(attitude_error_vector.x, -M_PI / 4, M_PI / 4) * _ahrs.get_gyro().z;', # 6
    'return MAX(ToDeg(_althold_lean_angle_max), AC_ATTITUDE_CONTROL_ANGLE_LIMIT_MIN) * 100.0f;', # 7
    'if ((vector_length > max_length) && is_positive(vector_length))', # 8
]

real_life_mutated_bugs = [
    '/*!! BUG !! */ float acro_level_mix = constrain_float(float(1-MAX(MAX(abs(roll_in), abs(pitch_in)), abs(yaw_in))/4500.0), 0, 1)*ahrs.cos_pitch();', # 0 mode acro
    '/*!! BUG !! */ //loiter_nav->clear_pilot_desired_acceleration();', # 1 mode auto
    '/*!! BUG !! */ //loiter_nav->clear_pilot_desired_acceleration();', # 2 mode auto
    '/*!! BUG !! */ //loiter_nav->clear_pilot_desired_acceleration();', # 3 mode rtl
    '/*!! BUG !! */ if (motors->get_spool_state() != AP_Motors::SpoolState::GROUND_IDLE)', # 4 full traces
    '/*!! BUG !! */ _rate_target_ang_vel.x += attitude_error_vector.y * _ahrs.get_gyro().z;', # 5 full traces
    '/*!! BUG !! */ _rate_target_ang_vel.y += -attitude_error_vector.x * _ahrs.get_gyro().z;', # 6
    '/*!! BUG !! */ return ToDeg(_althold_lean_angle_max) * 100.0f;', # 7
    '/*!! BUG !! */ if ((vector_length > max_length) && is_positive(max_length))', # 8
]

real_life_bug_group = [
    {'file':'ArduCopter/mode_acro.cpp',
    'indices':[0],
    'start':4000000,
    'lineno':[147]
    }, # 0
    {
        'file':'ArduCopter/mode_auto.cpp',
        'indices':[1,2],
        'start':5000000,
        'lineno':[836,933]
    }, # 1
    {   'file':'ArduCopter/mode_rtl.cpp',
        'indices':[3],
        'start':6000000,
        'lineno':[385],
    }, # 2
    {
        'file':'ArduCopter/motors.cpp',
        'indices':[4],
        'start':8000000,
        'lineno':[97]
    }, # 3
    {
        'file':'libraries/AC_AttitudeControl/AC_AttitudeControl.cpp',
        'indices':[5,6],
        'start':9000000,
        'lineno':[661]
    }, # 4
    {
        'file':'libraries/AC_AttitudeControl/AC_AttitudeControl.cpp',
        'indices':[7],
        'start':10000000,
        'lineno':[956]
    }, # 5
    {
        'file':'libraries/AC_AttitudeControl/AC_PosControl.cpp',
        'indices':[8],
        'start':11000000,
        'lineno':[1167]
    }, # 6
]
# # Single Bug
# for i, bug in enumerate(bug_group):
#     dest_dir = "%s%s%d/" % (ARDUPILOT_DIR, "bug_files/", i)
#     with open(ARDUPILOT_DIR+bug["file"]) as f:
#         file_data = f.read()
#         for idx in bug["indices"]:
#             file_data = file_data.replace(target_statements[idx], mutated_statements[idx])
#         new_file = "%s%s" % (dest_dir, bug["file"])
#         if not os.path.exists(os.path.dirname(new_file)):
#             os.makedirs(os.path.dirname(new_file))
#         with open(new_file, "w") as nf:
#             nf.write(file_data)
#         with open(dest_dir+"bug_location.txt", "w") as nf:
#             nf.write(str(bug["location"]))

def inject_bugs(selected_bugs_id,real_life=False):
    # selected_bugs = random.sample(target_statements,3)
    # selected_bugs = random.sample(range(0,len(target_statements)),3)
    # selected_bugs = [2,3]
    # print(selected_bugs)
    dest_dir = ARDUPILOT_DIR
    selected_bugs = []
    if real_life:
        targets = real_life_target_bugs
        mutated = real_life_mutated_bugs
        group = real_life_bug_group
    else:
        targets = target_statements
        mutated = mutated_statements
        group = bug_group
    for id in selected_bugs_id:
        print(id)
        selected_bugs.append(group[id])
    bug_info = ''
    for bug in selected_bugs:
        relative_file = open(ARDUPILOT_DIR+bug['file'])
        print(relative_file)
        relative_file_data = relative_file.read()
        relative_file.close()
        for idx in bug['indices']:
            if targets[idx] not in relative_file_data:
                print(targets[idx]+' not in '+bug['file'])
                exit(-1)
            relative_file_data = relative_file_data.replace(targets[idx],mutated[idx])
            bug_info += targets[idx] + '\n'
        new_file = '%s%s' % (dest_dir,bug['file'])
        if not os.path.exists(os.path.dirname(new_file)):
            os.makedirs(os.path.dirname(new_file))
        with open(new_file,'w') as f:
            f.write(relative_file_data)

if __name__ == '__main__':
    inject_bugs(range(0,7),True)