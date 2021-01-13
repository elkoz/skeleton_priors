import torch
import numpy as np

def angle_between_segments(prev_joint, joint, next_joint, rot_axis, leg):
    v1 = prev_joint - joint
    v2 = next_joint - joint
    #print(v1)
    #print(v2)
    cos_angle = v1 @ v2 / (torch.norm(v1) * torch.norm(v2) + torch.tensor(1e-6))
    angle = torch.acos(cos_angle)

    det = torch.det(torch.stack([rot_axis,v1,v2]))
    if det < 0:
        angle_corr = -angle
    else:
        angle_corr = angle

    return angle_corr

def calculate_yaw(coxa_origin, femur_pos, leg):
    next_joint = torch.cat([torch.reshape(coxa_origin[0], (1,)), femur_pos[1:]])
    z_axis = coxa_origin + torch.tensor([0,0,-1])
    rot_axis = torch.tensor([1,0,0])

    angle = angle_between_segments(z_axis, coxa_origin, next_joint, rot_axis, leg)

    return angle


def calculate_pitch(coxa_origin, femur_pos, leg):
    next_joint = torch.cat([femur_pos[0:1], torch.reshape(coxa_origin[1], (1,)), femur_pos[2:]])
    z_axis = coxa_origin + torch.tensor([0, 0, -1])
    rot_axis = torch.tensor([0, 1, 0])

    angle = angle_between_segments(z_axis, coxa_origin, next_joint, rot_axis, leg)

    return angle


def calculate_roll(coxa_origin, femur_pos, tibia_pos, r, leg):
    length = torch.norm(femur_pos - coxa_origin)

    if 'F_' in leg:
        prev_joint = torch.stack([torch.tensor(1), torch.tensor(0), -length])
        rot_axis = torch.tensor([0., 0., 1.])
    elif 'LM' in leg or 'LH' in leg:
        prev_joint = torch.stack([torch.tensor(0), torch.tensor(-1), -length])
        rot_axis = torch.tensor([0., 0., -1.])
    elif 'RM' in leg or 'RH' in leg:
        prev_joint = torch.stack([torch.tensor(0), torch.tensor(1), -length])
        rot_axis = torch.tensor([0., 0., 1.])

    curr_joint = torch.stack([torch.tensor(0), torch.tensor(0), -length])
    r_inv = torch.inverse(r)
    tibia = tibia_pos - coxa_origin
    next_joint = torch.cat([(r_inv @ tibia)[:-1], torch.reshape(-length, (1,))])

    angle = angle_between_segments(prev_joint, curr_joint, next_joint, rot_axis, leg)

    return angle


def calculate_angles(p_3d):
    leg_names = ['RH_leg', 'RM_leg', 'RF_leg', 'LH_leg', 'LM_leg', 'LF_leg']
    p_3d = torch.reshape(p_3d, (3, 30, -1))
    angles_dict = {}
    coxa_pos_dict = {}
    for i in range(len(leg_names)):
        leg = leg_names[i]
        joints = p_3d[:, i * 5:i * 5 + 5, :]
        angles_dict[leg] = {}
        factor_zero = torch.tensor(-np.pi)
        if 'F' in leg:
            rot_axis = torch.tensor([0, 1, 0])
        elif 'LM' in leg:
            rot_axis = torch.tensor([1, 0, 0])
        elif 'LH' in leg:
            rot_axis = torch.tensor([0, -1, 0])
        elif 'RM' in leg:
            rot_axis = torch.tensor([-1, 0, 0])
        elif 'RH' in leg:
            rot_axis = torch.tensor([0, -1, 0])

        coxa_positions, femur_positions, tibia_positions, tarsus_positions, claw_positions \
            = torch.transpose(torch.transpose(joints, 0, 1), 1, 2)
        coxa_pos = torch.mean(coxa_positions, axis=0)
        coxa_pos_dict[leg] = coxa_pos

        angles_dict[leg]['yaw'] = []
        angles_dict[leg]['pitch'] = []
        angles_dict[leg]['roll'] = []
        angles_dict[leg]['th_fe'] = []
        angles_dict[leg]['th_ti'] = []
        angles_dict[leg]['th_ta'] = []
        for k in range(coxa_positions.shape[0]):
            femur_pos = femur_positions[k]
            tibia_pos = tibia_positions[k]
            tarsus_pos = tarsus_positions[k]
            claw_pos = claw_positions[k]

            yaw = calculate_yaw(coxa_pos, femur_pos, leg)
            pitch = calculate_pitch(coxa_pos, femur_pos, leg)

            r = get_rot(torch.stack([yaw, pitch, torch.tensor(0.)]))
            roll = calculate_roll(coxa_pos, femur_pos, tibia_pos, r, leg)

            th_femur = angle_between_segments(coxa_pos, femur_pos, tibia_pos, rot_axis, leg)
            if th_femur > 0:
                th_femur = factor_zero + th_femur
            else:
                th_femur = factor_zero - th_femur

            th_tibia = angle_between_segments(femur_pos, tibia_pos, tarsus_pos, rot_axis, leg)
            if th_tibia < 0:
                th_tibia = th_tibia - factor_zero

            th_tarsus = angle_between_segments(tibia_pos, tarsus_pos, claw_pos, rot_axis, leg)
            if th_tarsus > 0:
                th_tarsus = factor_zero + th_tarsus
            else:
                th_tarsus = factor_zero - th_tarsus

            angles_dict[leg]['yaw'].append(yaw)
            angles_dict[leg]['pitch'].append(pitch)
            angles_dict[leg]['roll'].append(roll)
            angles_dict[leg]['th_fe'].append(th_femur)
            angles_dict[leg]['th_ti'].append(th_tibia)
            angles_dict[leg]['th_ta'].append(th_tarsus)

    return (angles_dict, coxa_pos_dict)

def deg2rad(a):
    return (a/180)*torch.tensor(np.pi)

def preprocess(angles_dict, offset=0):
    for leg in angles_dict.keys():
        for angle in angles_dict[leg].keys():
            angles_dict[leg][angle] = torch.stack(angles_dict[leg][angle])
    angles_dict['LF_leg']['roll'] = angles_dict['LF_leg']['roll'] + deg2rad(offset)
    angles_dict['RF_leg']['roll'] = angles_dict['RF_leg']['roll'] - deg2rad(offset)
    angles_dict['LM_leg']['roll'] = -(np.pi/2 + angles_dict['LM_leg']['roll']) - deg2rad(offset)
    angles_dict['LH_leg']['roll'] = -(np.pi/2 + angles_dict['LH_leg']['roll']) - deg2rad(offset)
    angles_dict['RM_leg']['roll'] = np.pi/2 + angles_dict['RM_leg']['roll'] - deg2rad(offset)
    angles_dict['RH_leg']['roll'] = np.pi/2 + angles_dict['RH_leg']['roll'] - deg2rad(offset)
    return angles_dict

def calculate_fk(angles_dict, leg_lengths, coxa_pos_dict):
    frames = len(angles_dict['LF_leg']['yaw'])
    fk = []
    for i in range(6):
        leg = leg_names[i]
        leg_angles = angles_dict[leg]
        coxa_pos = coxa_pos_dict[leg]
        l_coxa,l_femur,l_tibia,l_tarsus = leg_lengths[i, :]
        fe_init_pos = torch.reshape(torch.stack([torch.tensor([0]),torch.tensor([0]),-l_coxa]), (3,))
        ti_init_pos = torch.reshape(torch.stack([torch.tensor([0]),torch.tensor([0]),-l_femur]), (3,))
        ta_init_pos = torch.reshape(torch.stack([torch.tensor([0]),torch.tensor([0]),-l_tibia]), (3,))
        claw_init_pos = torch.reshape(torch.stack([torch.tensor([0]),torch.tensor([0]),-l_tarsus]), (3,))
        for frame in range(frames):
            roll_tr = torch.tensor(0)
            yaw_tr = torch.tensor(0)
            roll_ti = torch.tensor(0)
            yaw_ti = torch.tensor(0)
            roll_ta = torch.tensor(0)
            yaw_ta = torch.tensor(0)
            r1 = get_rot(torch.stack([leg_angles['yaw'][frame], leg_angles['pitch'][frame], leg_angles['roll'][frame]]))
            r2 = get_rot(torch.stack([yaw_tr, leg_angles['th_fe'][frame], roll_tr]))
            r3 = get_rot(torch.stack([yaw_ti, leg_angles['th_ti'][frame], roll_ti]))
            r4 = get_rot(torch.stack([yaw_ta, leg_angles['th_ta'][frame], roll_ta]))
            femur_pos = r1 @ fe_init_pos + coxa_pos
            tibia_pos = r1 @ r2 @ ti_init_pos + femur_pos
            tarsus_pos = r1 @ r2 @ r3 @ ta_init_pos + tibia_pos
            claw_pos = r1 @ r2 @ r3 @ r4 @ claw_init_pos + tarsus_pos
            fk.append(torch.stack([coxa_pos,femur_pos,tibia_pos,tarsus_pos,claw_pos]))
    fk = torch.stack(fk)
    fk = torch.reshape(fk, (6, -1, 5, 3))
    fk = torch.transpose(fk, 1, 3)
    fk = torch.transpose(fk, 0, 1)
    fk = torch.reshape(fk, (3, -1))
    return fk