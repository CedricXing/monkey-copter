# Randomly insert some bugs in two libaray files (AC_AttitudeControl/AC_PosControl.cpp and AC_WPNav/AC_WPNav.cpp)

import os
import random

bug_num_to_insert = 3

target_statements = [
    'linear_distance = _accel_z_cms / (2.0f * _p_pos_z.kP() * _p_pos_z.kP());', #0
    '_speed_cms = speed_cms;', #1
    'stopping_point.z = curr_pos_z + (linear_distance + curr_vel_z * curr_vel_z / (2.0f * _accel_z_cms));', #2
    'stopping_point.z = curr_pos_z - (linear_distance + curr_vel_z * curr_vel_z / (2.0f * _accel_z_cms))', #3
    '_pos_target.z = curr_alt + _leash_up_z;', #4
    '_pos_error.x = _pos_target.x - curr_pos.x;', #5
    '_pos_error.y = _pos_target.y - curr_pos.y;', #6
    '_accel_cms = accel_cmss;', #7
    '/*target*/_vel_desired.z = 0.0f;', #8
    '', #9
    '', #10
    '', #11 
    '', #12
    '', #13
    '', #14
    '', #15
    '', #16
    '', #17
    '', #18
    '', #19
    '', #20
    # nav
    '_track_accel = 0;', #11
    '_track_speed = 0;', #12
    '_pos_delta_unit.x = 0;', #13
    '_slow_down_dist = 0.0f;', #14
    'Vector3f dist_to_dest = (curr_pos - Vector3f(0,0,terr_offset)) - _destination;', #15
    '_track_leash_length = leash_z/pos_delta_unit_z;', #16
    '_limited_speed_xy_cms = MIN(_limited_speed_xy_cms,', #17
    '_limited_speed_xy_cms -= 2.0f * _track_accel * dt;', #18
    '_limited_speed_xy_cms = 0;', #19
    '_track_leash_length = WPNAV_LEASH_LENGTH_MIN;', #20
    '_track_leash_length = _pos_control.get_leash_xy()/pos_delta_unit_xy;', #21
    '_track_leash_length = leash_z/pos_delta_unit_z;', #22
    '_track_leash_length = MIN(leash_z/pos_delta_unit_z', #23
]

mutated_statements = [
    '/* BUG!! */ linear_distance = _accel_z_cms / (2.0f * _p_pos_z.kP());', #0
    '/* BUG!! */ _speed_cms = -speed_cms;', #1
    '/* BUG!! */ stopping_point.z =curr_pos_z - (linear_distance + curr_vel_z * curr_vel_z / (2.0f * _accel_z_cms));', #2
    '/* BUG!! */ stopping_point.z = curr_pos_z + (linear_distance + curr_vel_z * curr_vel_z / (2.0f * _accel_z_cms));', #3
    '/* BUG!! */ _pos_target.z = curr_alt - _leash_up_z;', #4
    '/* BUG!! */ _pos_error.x = curr_pos.x - _pos_target.x;', #5
    '/* BUG!! */ _pos_error.y = curr_pos.y - _pos_target.y;', #6
    '/* BUG!! */ _accel_cms = -accel_cmss;', #7
    '/*!! BUG !! */ ///*target*/_vel_desired.z = 0.0f;', #8
    '', #9
    '', #10
    '', #11
    '', #12
    '', #13
    '', #14
    '', #15
    '', #16
    '', #17
    '', #18
    '', #19
    '', #20
    #nav
    '/* BUG!! */ //_track_accel = 0;', #21
    '/* BUG!! */ //_track_speed = 0;', #22
    '/* BUG!! */ return false;', #23
    '/* BUG!! */ //_slow_down_dist = 0.0f;', #24
    '/* BUG!! */ Vector3f dist_to_dest = (curr_pos + Vector3f(0,0,terr_offset)) - _destination;', #25
    '/* BUG!! */ //_track_leash_length = leash_z/pos_delta_unit_z;', #26
    '/* BUG!! */ _limited_speed_xy_cms = MAX(_limited_speed_xy_cms,', #27
    '/* BUG!! */ _limited_speed_xy_cms += 2.0f * _track_accel * dt;', #28
    '/* BUG!! */ //_limited_speed_xy_cms = 0;', #29
    '/* BUG!! */ //_track_leash_length = WPNAV_LEASH_LENGTH_MIN;', #30
    '/* BUG!! */ //_track_leash_length = _pos_control.get_leash_xy()/pos_delta_unit_xy;', #31
    '/* BUG!! */ //_track_leash_length = leash_z/pos_delta_unit_z;', #32
    '/* BUG!! */ //_track_leash_length = MIN(leash_z/pos_delta_unit_z', #33
]

bug_group = [
    {
        "file": "libraries/AC_AttitudeControl/AC_PosControl.cpp",
        "indices": [0],
        "lineno" : [430],
     }, #0
     {
         "file": "libraries/AC_AttitudeControl/AC_PosControl.cpp",
        "indices": [1],
        "lineno" : [621],
    }, #1
    {
        "file": "libraries/AC_AttitudeControl/AC_PosControl.cpp",
        "indices": [2,3],
        "lineno" : [427,432],
    }, #2
    {
        "file": "libraries/AC_AttitudeControl/AC_PosControl.cpp",
        "indices": [4],
        "lineno" : [514],
    }, #3
    {
        "file": "libraries/AC_AttitudeControl/AC_PosControl.cpp",
        "indices": [5,6],
        "lineno" : [981],
    }, #4
    {
        "file": "libraries/AC_AttitudeControl/AC_PosControl.cpp",
        "indices": [7],
        "lineno" : [611],
    }, #5
    {
        "file":"libraries/AC_AttitudeControl/AC_PosControl.cpp",
        "indices":[8],
        'lineno':[287]
    }, #6
    {
        "file": "libraries/AC_WPNav/AC_WPNav.cpp",
        "indices": [21,22],
        "lineno" : [545],
    }, #7
    {
        "file": "libraries/AC_WPNav/AC_WPNav.cpp",
        "indices": [23],
        "lineno" : [237],
    }, #8
    {
        "file": "libraries/AC_WPNav/AC_WPNav.cpp",
        "indices": [24],
        "lineno" : [1004],
    }, #9
    {
        "file": "libraries/AC_WPNav/AC_WPNav.cpp",
        "indices": [25],
        "lineno" : [442],
    }, #10
    {
        "file": "libraries/AC_WPNav/AC_WPNav.cpp",
        "indices": [26],
        "lineno" : [553],
    }, #11
    {
        "file": "libraries/AC_WPNav/AC_WPNav.cpp",
        "indices": [27],
        "lineno" : [399],
    }, #12
    {
        "file": "libraries/AC_WPNav/AC_WPNav.cpp",
        "indices": [28],
        "lineno" : [414],
    }, #13
    {
        "file": "libraries/AC_WPNav/AC_WPNav.cpp",
        "indices": [29],
        "lineno" : [381],
    }, #14
    {
        "file": "libraries/AC_WPNav/AC_WPNav.cpp",
        "indices": [30,31,32],
        "lineno" : [545,549,553],
    }, #15
]

real_life_target_bugs = [
    'float acro_level_mix = constrain_float(1-float(MAX(MAX(abs(roll_in), abs(pitch_in)), abs(yaw_in))/4500.0), 0, 1)*ahrs.cos_pitch();', #0
    'leash_length = POSCONTROL_LEASH_LENGTH_MIN', #1
    'loiter_nav->clear_pilot_desired_acceleration();', #2
    'if (motors->get_spool_state() > AP_Motors::SpoolState::GROUND_IDLE)', #3
    '_rate_target_ang_vel.x += constrain_float(attitude_error_vector.y, -M_PI / 4, M_PI / 4) * _ahrs.get_gyro().z;', #4
    '_rate_target_ang_vel.y += -constrain_float(attitude_error_vector.x, -M_PI / 4, M_PI / 4) * _ahrs.get_gyro().z;', #5
    'return MAX(ToDeg(_althold_lean_angle_max), AC_ATTITUDE_CONTROL_ANGLE_LIMIT_MIN) * 100.0f;', #6
    'if ((vector_length > max_length) && is_positive(vector_length))', #7
    'vector_x *= (max_length / vector_length);', #8
    'vector_y *= (max_length / vector_length);', #9
]

real_life_mutated_bugs = [
    '/*!! BUG !! */ float acro_level_mix = constrain_float(float(1-MAX(MAX(abs(roll_in), abs(pitch_in)), abs(yaw_in))/4500.0), 0, 1)*ahrs.cos_pitch();', #0
    '/* BUG!! */ //leash_length = POSCONTROL_LEASH_LENGTH_MIN', #1
    '/*!! BUG !! */ //loiter_nav->clear_pilot_desired_acceleration();', #2
    '/*!! BUG !! */ if (motors->get_spool_state() != AP_Motors::SpoolState::GROUND_IDLE)', #3
    '/*!! BUG !! */ _rate_target_ang_vel.x += attitude_error_vector.y * _ahrs.get_gyro().z;', #4
    '/*!! BUG !! */ _rate_target_ang_vel.y += -attitude_error_vector.x * _ahrs.get_gyro().z;', #5
    '/*!! BUG !! */ return ToDeg(_althold_lean_angle_max) * 100.0f;', #6
    '/*!! BUG !! */ if ((vector_length > max_length) && is_positive(max_length))', #7
    '/* BUG!! */ vector_x = (max_length / vector_length);', #8
    '/* BUG!! */ vector_y = (max_length / vector_length);', #9
]

real_life_bug_group = [
    {
        'file':'ArduCopter/mode_acro.cpp',
        'indices':[0],
        'lineno':[147]
    }, #0
    {
        "file": "libraries/AC_AttitudeControl/AC_PosControl.cpp",
        "indices": [1],
        "lineno" : [1118],
     }, #1
    {
        'file':'ArduCopter/motors.cpp',
        'indices':[3],
        'lineno':[97]
    }, #2
    {
        'file':'libraries/AC_AttitudeControl/AC_AttitudeControl.cpp',
        'indices':[4,5],
        'lineno':[661]
    }, #3
    {
        'file':'libraries/AC_AttitudeControl/AC_AttitudeControl.cpp',
        'indices':[6],
        'lineno':[954]
    }, #4
    {
        'file':'libraries/AC_AttitudeControl/AC_PosControl.cpp',
        'indices':[7],
        'lineno':[1167]
    }, #5
    {
        "file": "libraries/AC_AttitudeControl/AC_PosControl.cpp",
        "indices": [8,9],
        "lineno" : [1167],
    }, #6
]

def inject_bugs(selected_bugs_id,config):
    dest_dir = config['root_dir']
    selected_bugs = []
    if config['real_life'] == 'True':
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
        relative_file = open(config['root_dir']+bug['file'])
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
