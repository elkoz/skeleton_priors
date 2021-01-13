import torch
import numpy as np
from df3dPostProcessing.df3dPostProcessing.utils import utils_angles
import cv2
from itertools import combinations
import einops
import scipy

def get_3d(angles, data_dict, begin=0, end=0, ik_angles=False, offset=0):
    ''' The function used to generate synthetic data'''
    if end == 0:
        end = len(angles['LF_leg']['yaw'])

    order = ['RH_leg', 'RM_leg', 'RF_leg', 'LH_leg', 'LM_leg', 'LF_leg']
    angles_rev = {leg: angles[leg] for leg in order}
    positions = []

    for frame in range(begin, end):
        positions.append(np.zeros((30, 3)))
        i = 0
        for name, leg in angles_rev.items():

            pos_3d = utils_angles.calculate_forward_kinematics(name, frame, leg, data_dict, extraDOF={},
                                                                   ik_angles=ik_angles, offset=offset).transpose()

            positions[-1][i * 5: (i + 1) * 5, :] = pos_3d.T
            i += 1
    positions = torch.tensor(positions, dtype=torch.float32)
    positions = torch.transpose(positions, 0, 2)
    return positions

def rot2eul(R):
    ''' Extracts angles from a rotation matrix'''
    beta = -np.arcsin(R[2,0])
    alpha = np.arctan2(R[2,1]/np.cos(beta),R[2,2]/np.cos(beta))
    gamma = np.arctan2(R[1,0]/np.cos(beta),R[0,0]/np.cos(beta))
    return torch.tensor((alpha, beta, gamma))

def get_rot(angles):
    ''' Transforms angles into a rotation matrix'''
    a, b, c = angles
    Rx = torch.stack([torch.tensor(1), torch.tensor(0), torch.tensor(0), torch.tensor(0), torch.cos(a),
                      -torch.sin(a), torch.tensor(0), torch.sin(a), torch.cos(a)])
    Ry = torch.tensor([torch.cos(b), torch.tensor(0), torch.sin(b), torch.tensor(0),
                       torch.tensor(1), torch.tensor(0), -torch.sin(b), torch.tensor(0), torch.cos(b)])
    Rz = torch.tensor([torch.cos(c), -torch.sin(c), torch.tensor(0), torch.sin(c), torch.cos(c), torch.tensor(0),
                       torch.tensor(0), torch.tensor(0), torch.tensor(1)])
    Rx = torch.reshape(Rx, (3, 3))
    Ry = torch.reshape(Ry, (3, 3))
    Rz = torch.reshape(Rz, (3, 3))
    return Rx @ Ry @ Rz

def intr2par(intr):
    ''' Converts intrinsic matrix into camera parameters'''
    intr = torch.tensor(intr)
    cx = intr[0, 2]
    cy = intr[1, 2]
    fx = intr[0, 0]
    a = torch.atan(-fx/intr[0, 1])
    fy = torch.abs(intr[1, 1] * torch.sin(a))
    focus = torch.tensor([fx, fy])
    center = torch.tensor([cx, cy])
    return (center, focus, a)

def pars_from_dict(file):
    ''' Extracts all camera parameters from a json file'''
    center = []
    focus = []
    skew = []
    tvec = []
    angles = []
    coef = []
    for i in [0, 1, 2, 4, 5, 6]:
        angles_i = rot2eul(file[i]['R'])
        center_i, focus_i, skew_i = intr2par(file[i]['intr'])
        tvec_i = torch.tensor(file[i]['tvec'])
        coef_i = torch.tensor(file[i]['distort'])
        angles.append(angles_i)
        center.append(center_i)
        focus.append(focus_i)
        skew.append(skew_i)
        tvec.append(tvec_i)
        coef.append(coef_i)
    center = torch.stack(center).float()
    center.requires_grad = True
    focus = torch.stack(focus).float()
    focus.requires_grad = True
    skew = torch.stack(skew).float() - torch.tensor(np.pi/2)
    skew.requires_grad = True
    tvec = torch.stack(tvec).float()
    tvec.requires_grad = True
    angles = torch.stack(angles).float()
    angles.requires_grad = True
    coef = torch.stack(coef).float()
    coef = torch.transpose(coef, 0, 1).unsqueeze(2)
    coef.requires_grad = True
    angles_between = torch.tensor([0., 3*np.pi/2, 0.]).float()
    angles_between.requires_grad = True
    return (center, focus, skew, tvec, angles, coef, angles_between)

def triangulate(p, init_angles, angles, focus, center, skew, tvec):
    ''' Triangulates synthetic data'''
    matrices = []
    init_rvec = get_rot(init_angles)
    rvec = torch.stack([get_rot(angle) @ init_rvec for angle in angles], 0)
    for i in range(6):
        fx, fy = focus[i]
        cx, cy = center[i]
        a = torch.tensor(np.pi/2) + skew[i]
        K = torch.stack([fx, -fx*torch.cos(a)/torch.sin(a), cx, torch.tensor(0), torch.tensor(0), fy/torch.sin(a), cy,
                         torch.tensor(0), torch.tensor(0), torch.tensor(0), torch.tensor(1), torch.tensor(0)])
        K = torch.reshape(K, (3, 4))
        M = torch.cat((rvec[i], torch.reshape(tvec[i], (3, 1))), axis = 1)
        M = torch.cat((M, torch.tensor([[0, 0, 0, 1]])))
        matrices.append((K @ M).detach().numpy())
    points_list = []
    for i in combinations(range(6), 2):
        points = cv2.triangulatePoints(matrices[0], matrices[1], np.float32(p[0]), np.float32(p[1]))
        points = points[:-1, :] / points[-1:, :]
        points_list.append(points)
    points = np.mean(np.stack(points_list), axis = 0)
    points = torch.tensor(points)
    return points


def triangulate_real(p, angles_between, angles, focus, center, skew, tvec):
    ''' Triangulates real data'''
    matrices = []
    mid_rvec = get_rot(angles_between)
    rvec = torch.stack([get_rot(angle) for angle in angles[:3]] + \
                       [get_rot(angle) @ mid_rvec for angle in angles[3:]], 0)
    for i in range(6):
        fx, fy = focus[i]
        cx, cy = center[i]
        a = skew[i] + torch.tensor(np.pi / 2)
        K = torch.stack(
            [fx, -fx * torch.cos(a) / torch.sin(a), cx, torch.tensor(0), torch.tensor(0), fy / torch.sin(a), cy,
             torch.tensor(0), torch.tensor(0), torch.tensor(0), torch.tensor(1), torch.tensor(0)])
        K = torch.reshape(K, (3, 4))
        M = torch.cat((rvec[i], torch.reshape(tvec[i], (3, 1))), axis=1)
        M = torch.cat((M, torch.tensor([[0, 0, 0, 1]])))
        matrices.append((K @ M).detach().numpy())
    points_list = []
    for i, j in combinations(range(3), 2):
        points = cv2.triangulatePoints(matrices[i], matrices[j], np.float32(p[i]), np.float32(p[j]))
        points = points[:-1, :] / points[-1:, :]
        points_list.append(points)
    res = np.mean(np.stack(points_list), axis=0)
    points_list = []
    for i, j in combinations(range(3, 6), 2):
        points = cv2.triangulatePoints(matrices[i], matrices[j], np.float32(p[i]), np.float32(p[j]))
        points = points[:-1, :] / points[-1:, :]
        points_list.append(points)
    res += np.mean(np.stack(points_list), axis=0)
    res = torch.tensor(res)
    return res

def distortion(ps, coef):
    ''' Adds distortion to projections ontp camera planes'''
    x, y = ps[:, 0, :], ps[:, 1, :]
    k1, k2, k3, p1, p2 = coef
    r = torch.sqrt(x**2 + y**2)
    x = x*(1 + k1*r**2 + k2*r**4 + k3*r**6) + 2*p1*x*y + p2
    y = y*(1 + k1*r**2 + k2*r**4 + k3*r**6) + p1*(r**2 + 2*y**2) + 2*p2*x*y
    return torch.stack((x, y), axis = 1)

def get_circle():
    ''' Generates angles for the synthetic camera positions'''
    zero_angle = torch.tensor([0.0, 0.0, 0.0])
    rot_angles = [zero_angle + i*torch.tensor([0., np.pi/3, 0.]) for i in range(6)]
    return torch.stack(rot_angles, 0)

def projection(p_3d, vector, focus, skew, angles, init_angles, tvec, coef):
    ''' Projects synthetic 3D positions ontp camera planes'''
    num_cam = vector.shape[0]
    KMs = []
    init_rvec = get_rot(init_angles)
    rvec = torch.stack([get_rot(angle) @ init_rvec for angle in angles], 0)
    for i in range(num_cam):
        num_cam = 6
        fx, fy = focus[i]
        cx, cy = vector[i]
        a = torch.tensor(np.pi/2) + skew[i]
        K = torch.stack([fx, -fx*torch.cos(a)/torch.sin(a), cx, torch.tensor(0), torch.tensor(0), fy/torch.sin(a), cy,
                         torch.tensor(0), torch.tensor(0), torch.tensor(0), torch.tensor(1), torch.tensor(0)])
        K = torch.reshape(K, (3, 4))
        M = torch.cat((rvec[i], torch.reshape(tvec[i], (3, 1))), axis = 1)
        M = torch.cat((M, torch.tensor([[0, 0, 0, 1]])))
        KMs.append(K @ M)
    KMs = torch.stack(KMs, 0)
    p_2d = KMs @ torch.cat((p_3d, torch.ones((1, p_3d.shape[1]))))
    p_2d = torch.true_divide(p_2d[:, :-1, :], p_2d[:, -1:, :])
    p_2d = distortion(p_2d, coef)
    return p_2d

def projection_real(p_3d, vector, focus, skew, angles, angles_between, tvec, coef):
    ''' Projects real 3D positions ontp camera planes'''
    num_cam = vector.shape[0]
    KMs = []
    mid_rvec = get_rot(angles_between)
    rvec = torch.stack([get_rot(angle) for angle in angles[:3]] + \
                   [get_rot(angle) @ mid_rvec for angle in angles[3:]], 0)
    for i in range(num_cam):
        num_cam = 6
        fx, fy = focus[i]
        cx, cy = vector[i]
        a = torch.tensor(np.pi/2) + skew[i]
        K = torch.stack([fx, -fx*torch.cos(a)/torch.sin(a), cx, torch.tensor(0), torch.tensor(0), fy/torch.sin(a), cy,
                         torch.tensor(0), torch.tensor(0), torch.tensor(0), torch.tensor(1), torch.tensor(0)])
        K = torch.reshape(K, (3, 4))
        M = torch.cat((rvec[i], torch.reshape(tvec[i], (3, 1))), axis = 1)
        M = torch.cat((M, torch.tensor([[0, 0, 0, 1]])))
        KMs.append(K @ M)
    KMs = torch.stack(KMs, 0)
    p_2d = KMs @ torch.cat((p_3d, torch.ones((1, p_3d.shape[1]))))
    p_2d = torch.true_divide(p_2d[:, :-1, :], p_2d[:, -1:, :])
    p_2d = distortion(p_2d, coef)
    return p_2d


def on_start(start, delta, x):
    ''' Helper function for interval_func'''
    x_p = torch.true_divide(x - start + delta, delta) - 1
    return 3 * (-x_p) ** 2 - 2 * (-x_p) ** 3


def on_end(end, delta, x):
    ''' Helper function for interval_func'''
    x_p = torch.true_divide(x - end, delta)
    return 3 * x_p ** 2 - 2 * x_p ** 3


def interval_func(start, end, delta):
    ''' Generates smoothstep-like functions for angle-based losses'''
    def func(x):
        res = torch.zeros(x.shape)
        res[(x <= start - delta) | (x >= end + delta)] = 1
        res[(x < start) & (x > start - delta)] = on_start(start, delta, x)[(x < start) & (x > start - delta)]
        res[(x > end) & (x < end + delta)] = on_end(end, delta, x)[(x > end) & (x < end + delta)]
        return res

    return func


def get_func(intervals, delta):
    ''' Generates smoothstep-like functions for angle-based losses (may be used with multiple intervals)'''
    def func(x):
        r = torch.zeros(x.shape)
        for start, end in intervals:
            r += interval_func(start, end, delta)(x)
        return r - len(intervals) + 1

    return func


def construct(coxa, angles, legs):
    ''' Constructs fk model from estimated parameters'''
    meds = (coxa[:, :3] + coxa[:, 3:]) / 2
    a = meds[:, 2] - meds[:, 0]
    n = -torch.cross(a, torch.tensor([0., 1., 0.]))
    n = torch.stack([n[0], torch.tensor(0), n[2]])
    n = n / torch.norm(n)

    cos = n @ torch.tensor([-1, 0., 0])
    n_angle_r = torch.acos(cos) * 180 / np.pi
    n_angle_l = n_angle_r + 180

    init = torch.zeros((angles.shape[-1], 4, 6, 3))
    init[:, :, :, 1] = (init[:, :, :, 1] + legs.T)
    femurs, tibias, tarsuses, claws = [init.T[:, :, i:i + 1, :] for i in range(4)]

    rot_roll = [torch.stack([get_rot_y(n_angle_r + a) for a in angles_leg[0]])
                for angles_leg in angles[:3]] + \
               [torch.stack([get_rot_y(-a + n_angle_l) for a in angles_leg[0]])
                for angles_leg in angles[3:]]
    rot_roll = torch.stack(rot_roll)

    rot_roll_2 = [torch.stack([get_rot_x(a) for a in angles_leg[-1]])
                  for angles_leg in angles]
    rot_roll_2 = torch.stack(rot_roll_2)

    rotations = []
    rot_tc = [torch.stack([get_rot_z(a) for a in angles_leg[1]])
              for angles_leg in angles]
    rotations.append(torch.stack(rot_tc))
    rot_cf = [torch.stack([get_rot_z(a) for a in angles_leg[2]])
              for angles_leg in angles]
    rot_cf = torch.stack(rot_cf)
    rotations.append(torch.einsum('ilpk,ilkj->ilpj', rot_cf, rotations[-1]))
    rot_ft = [torch.stack([get_rot_z(a) for a in angles_leg[3]])
              for angles_leg in angles]
    rot_ft = torch.stack(rot_ft)
    rotations.append(torch.einsum('ilpk,ilkj->ilpj', rot_ft, rotations[-1]))
    rot_tt = [torch.stack([get_rot_z(a) for a in angles_leg[4]])
              for angles_leg in angles]
    rot_tt = torch.stack(rot_tt)
    rotations.append(torch.einsum('ilpk,ilkj->ilpj', rot_tt, rotations[-1]))

    for i in range(4):
        rotations[i] = torch.einsum('ilpk,ilkj->ilpj', rot_roll_2, rotations[i])

    for i in range(4):
        rotations[i] = torch.einsum('ilpk,ilkj->ilpj', rot_roll, rotations[i])

    femurs = torch.einsum('iljk,kiml->jiml', rotations[0], femurs)
    tibias = torch.einsum('iljk,kiml->jiml', rotations[1], tibias)
    tarsuses = torch.einsum('iljk,kiml->jiml', rotations[2], tarsuses)
    claws = torch.einsum('iljk,kiml->jiml', rotations[3], claws)

    points = einops.repeat(coxa, 'm n -> m n k l', k=5, l=angles.shape[-1])
    points[:, :, 1:, :] += femurs
    points[:, :, 2:, :] += tibias
    points[:, :, 3:, :] += tarsuses
    points[:, :, 4:, :] += claws

    return points.reshape((3, -1))

def pars_from_points(fly_points):
    ''' Estimates model parameters from 3D data'''
    p_3d = torch.reshape(fly_points, (3, 6, 5, -1))
    coxa = torch.mean(p_3d[:, :, 0, :], axis = -1)
    legs = torch.norm(p_3d[:, :, 1:, :] - p_3d[:, :, :-1, :], dim=0)
    legs = torch.mean(legs, dim=-1)
    a = calculate_angles_2(fly_points).unsqueeze(1)
    b = calculate_angles_3(fly_points)
    angles = torch.cat([a, b], dim=1)
    c = calculate_angles_4(fly_points, angles, legs).unsqueeze(1)
    angles = torch.cat([angles, c], dim=1)
    return (coxa, angles, legs)

def calculate_angles_3(rand):
    p_3d = torch.reshape(rand, (3, 6, 5, -1))
    legs = p_3d[:, :, 1:, :] - p_3d[:, :, :-1, :]
    legs = torch.cat([einops.repeat(torch.tensor([0, 1., 0]), 'm -> m k l n', k=6, l=1, n=legs.shape[-1]), legs], dim=2)
    norms = torch.norm(legs, dim=0)
    down = norms[:, :-1, :] * norms[:, 1:, :]
    up = torch.einsum('kilm,kilm->ilm', legs[:, :, :-1, :], legs[:, :, 1:, :])
    cos = torch.true_divide(up, down)
    angles = torch.acos(cos) * 180 / np.pi
    cross = torch.cross(legs[:, :, :-1, :], legs[:, :, 1:, :], dim=0)
    plus_direction = torch.cross(legs[:, :, 1, :], legs[:, :, 2, :])
    direction = torch.sign(torch.einsum('kimj,kij->imj', cross, plus_direction))
    angles = angles * direction

    return angles

def calculate_angles_2(rand):
    p_3d = torch.reshape(rand, (3, 6, 5, -1))
    meds = (p_3d[:, :3, 0, :] + p_3d[:, 3:, 0, :])/2
    a = torch.mean(meds[:, 2, :] - meds[:, 0, :], dim=-1)
    n = -torch.cross(a, torch.tensor([0., 1., 0.]))
    n = torch.stack([n[0], torch.tensor(0.), n[2]])
    n = n/torch.norm(n)
    femur = p_3d[:, :, 2, :] - p_3d[:, :, 0, :]
    femur = torch.stack([femur[0, :, :], torch.zeros(femur.shape[1:]), femur[2, :, :]])
    up = torch.einsum('k,kil->il', n, femur)
    down = torch.norm(femur, dim=0)
    cos = torch.true_divide(up, down)
    angles = torch.acos(cos) * 180/np.pi
    angles[angles > 90] = angles[angles > 90] - 180
    cross = torch.cross(femur, einops.repeat(n, 'm -> m k n', k=6, n=femur.shape[-1]), dim=0)[1, :, :]
    angles = -torch.sign(cross)*angles
    angles = torch.cat([angles[:3, :], -angles[3:, :]], dim=0)
    return angles

def calculate_angles_4(fly_points, j_angles, legs):
    p_3d = torch.reshape(fly_points, (3, 6, 5, -1))
    meds = (p_3d[:, :3, 0, :] + p_3d[:, 3:, 0, :])/2
    a = torch.mean(meds[:, 2, :] - meds[:, 0, :], dim=-1)
    n = -torch.cross(a, torch.tensor([0., 0., 1.]))
    n = torch.stack([n[0], n[1], torch.tensor(0)])
    n = n/torch.norm(n)
    cos = n @ torch.tensor([-1, 0., 0])
    n_angle_r = torch.acos(cos) * 180/np.pi
    n_angle_l = n_angle_r + 180
    rot_roll = [torch.stack([get_rot_z(a+n_angle_l) for a in angles_leg[0]])
              for angles_leg in j_angles[:3]] + \
                [torch.stack([get_rot_z(n_angle_r-a) for a in angles_leg[0]])
              for angles_leg in j_angles[3:]]
    rot_roll = torch.stack(rot_roll)
    a = torch.einsum('iljk,k->jil', rot_roll, a)
    legs = p_3d[:, :, 1:3, :] - p_3d[:, :, :2, :]
    cross = torch.cross(legs[:, :, 0, :], legs[:, :, 1, :], dim=0)
    up = torch.einsum('ijk,ijk->jk', cross, a)
    down = torch.norm(a) * torch.norm(cross, dim=0)
    sin = torch.true_divide(up, down)
    angles = torch.asin(sin) * 180/np.pi
    return angles

def procrust(points, fly_points):
    ''' Procrustes distance'''
    err = scipy.spatial.procrustes(fly_points.T, points.detach().numpy().T)[2]
    return err

def get_rot_y(b):
    b = b*np.pi/180
    Ry = torch.tensor([torch.cos(b), torch.tensor(0), torch.sin(b), torch.tensor(0),
                       torch.tensor(1), torch.tensor(0), -torch.sin(b), torch.tensor(0), torch.cos(b)])
    Ry = torch.reshape(Ry, (3, 3))
    return Ry

def get_rot_z(c):
    c = c*np.pi/180
    Rz = torch.tensor([torch.cos(c), -torch.sin(c), torch.tensor(0), torch.sin(c), torch.cos(c), torch.tensor(0),
                       torch.tensor(0), torch.tensor(0), torch.tensor(1)])
    Rz = torch.reshape(Rz, (3, 3))
    return Rz

def get_rot_x(a):
    a = a*np.pi/180
    Rx = torch.stack([torch.tensor(1), torch.tensor(0), torch.tensor(0), torch.tensor(0), torch.cos(a),
                      -torch.sin(a), torch.tensor(0), torch.sin(a), torch.cos(a)])
    Rx = torch.reshape(Rx, (3, 3))
    return Rx


def initialize(file, start=0, end=30):
    ''' Get 2D and 3D points from json file'''
    R1 = file[0]['R']
    t1 = file[0]['tvec']
    intr1 = file[0]['intr']
    R2 = file[1]['R']
    t2 = file[1]['tvec']
    intr2 = file[1]['intr']
    R3 = file[4]['R']
    t3 = file[4]['tvec']
    intr3 = file[4]['intr']
    R4 = file[5]['R']
    t4 = file[5]['tvec']
    intr4 = file[5]['intr']

    p_real = file['points2d']
    p_real = np.delete(p_real, 3, 0)
    p_real = np.delete(p_real, [15, 16, 17, 18, 34, 35, 36, 37], 2)
    p_real = p_real[:, start:end, :, :]
    p_real = np.transpose(p_real, (0, 3, 2, 1))
    p_real = p_real.reshape((6, 2, -1))

    M1 = intr1 @ np.hstack([R1, np.expand_dims(t1, 1)])
    M2 = intr2 @ np.hstack([R2, np.expand_dims(t2, 1)])
    points_0 = cv2.triangulatePoints(M1, M2, p_real[0], p_real[1])
    points_0 = points_0[:-1] / points_0[-1]

    M3 = intr3 @ np.hstack([R3, np.expand_dims(t3, 1)])
    M4 = intr4 @ np.hstack([R4, np.expand_dims(t4, 1)])
    points_1 = cv2.triangulatePoints(M3, M4, p_real[3], p_real[4])
    points_1 = points_1[:-1] / points_1[-1]

    res = np.zeros((3, 900))
    res[:, p_real[0, 0, :] != 0] = (points_0)[:, p_real[0, 0, :] != 0]
    res[:, p_real[3, 0, :] != 0] = (points_1)[:, p_real[3, 0, :] != 0]
    res = torch.tensor(res).float()
    p_real = torch.tensor(p_real).float()
    return (res, p_real)


