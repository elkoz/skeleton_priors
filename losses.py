import torch
from utils import projection, projection_real, construct, get_func
from fk import calculate_angles, preprocess, calculate_fk
import numpy as np
import einops


def repr_error(p, p_3d, vector, focus, skew, angles, init_angles, tvec, coef, scale = 0):
    ''' Classic reprojection error for synthetic data'''
    p_2d = projection(p_3d, vector, focus, skew, angles, init_angles, tvec, coef)
    error = torch.mean(torch.sqrt(torch.sum((p_2d-p)**2, axis = 1)))
    return error

def repr_error_real(p, p_3d, vector, focus, skew, angles, tvec, coef, scale = 0):
    ''' Classic reprojection error for real data'''
    p_2d = projection_real(p_3d, vector, focus, skew, angles, tvec, coef)
    error_arr = ((p_2d-p)**2) * (p != 0)
    error = torch.mean(torch.sqrt(torch.sum(error_arr, axis = 1) + 1e-10)) * 2
    return error

def repr_error_new(p, coxa, joint_angles, legs, vector, focus, skew, angles, tvec, coef, scale = 0):
    ''' Classic reprojection error for real data (model parameters optimization)'''
    p_3d = construct(coxa, joint_angles, legs)
    p_2d = projection_real(p_3d, vector, focus, skew, angles, tvec, coef)
    error_arr = ((p_2d-p)**2) * (p != 0)
    error = torch.mean(torch.sqrt(torch.sum(error_arr, axis = 1) + 1e-10)) * 2
    return error

def relaxed_repr_error(p, p_3d, vector, focus, skew, angles, init_angles, tvec, coef, scale):
    ''' Relaxed reprojection error for synthetic data'''
    p_2d = projection(p_3d, vector, focus, skew, angles, init_angles, tvec, coef)
    norm = torch.distributions.normal.Normal(0, scale)
    e = torch.sqrt(torch.sum((p_2d - p)**2, axis = 1))
    n_e = torch.exp(norm.log_prob(e))
    n_zero = torch.exp(norm.log_prob(0))
    error = torch.mean((1 - n_e/n_zero)*e)
    return error

def relaxed_repr_error_new(p, coxa, joint_angles, legs, vector, focus, skew, angles, tvec, coef, scale):
    ''' Relaxed reprojection error for real data (model parameters optimization)'''
    p_3d= construct(coxa, joint_angles, legs)
    p_2d = projection_real(p_3d, vector, focus, skew, angles, tvec, coef)
    error_arr = ((p_2d - p)**2) * (p != 0)
    norm = torch.distributions.normal.Normal(0, scale)
    e = torch.sqrt(torch.sum(error_arr + 1e-10, axis = 1))
    n_e = torch.exp(norm.log_prob(e))
    n_zero = torch.exp(norm.log_prob(0))
    error = torch.mean((1 - n_e/n_zero)*e) * 2
    return error

def relaxed_repr_error_real(p, p_3d, vector, focus, skew, angles, tvec, coef, scale):
    ''' Relaxed reprojection error for real data'''
    p_2d = projection_real(p_3d, vector, focus, skew, angles, tvec, coef)
    error_arr = ((p_2d - p)**2) * (p != 0)
    norm = torch.distributions.normal.Normal(0, scale)
    e = torch.sqrt(torch.sum(error_arr + 1e-10, axis = 1))
    n_e = torch.exp(norm.log_prob(e))
    n_zero = torch.exp(norm.log_prob(0))
    error = torch.mean((1 - n_e/n_zero)*e) * 2
    return error

def leg_length_error(p_3d):
    '''Leg length error (model parameters optimization)'''
    p_3d = torch.reshape(p_3d, (3, 6, 5, -1))
    ll = torch.norm(p_3d[:, :, 1: :] - p_3d[:, :, :-1, :], dim = 0)
    fp = torch.reshape(p_3d, (3, 6, 5, -1))
    fp = torch.mean(fp, axis=3)
    fp = torch.unsqueeze(fp, 3)
    leg_lengths = torch.norm(fp[:, :, 1:, :] - fp[:, :, :-1, :], dim = 0)
    error = torch.mean((ll - leg_lengths)**2)
    return error

def leg_length_error_new(legs, true_legs):
    '''Leg length error'''
    error_arr = true_legs-legs
    error = torch.mean(error_arr**2)
    return error

def smooth_error(p_3d):
    ''' Smoothness error'''
    p_3d = torch.reshape(p_3d, (3, 30, -1))
    u = p_3d[:, :, 4:] - 2*p_3d[:, :, 3:-1] + 2*p_3d[:, :, 1:-3] - p_3d[:, :, :-4]
    error = torch.mean(u**2)*3
    return error

def smooth_error_new(coxa, angles, legs):
    ''' Smoothness error (model parameters optimization)'''
    p_3d = construct(coxa, angles, legs)
    p_3d = torch.reshape(p_3d, (3, 30, -1))
    u = p_3d[:, :, 4:] - 2*p_3d[:, :, 3:-1] + 2*p_3d[:, :, 1:-3] - p_3d[:, :, :-4]
    error = torch.mean(u**2)*3
    return error

def fk_error(p_3d, leg_lengths):
    ''' Forward kinematics error for synthetic data'''
    angles_dict, coxa_pos_dict = calculate_angles(p_3d)
    angles_dict = preprocess(angles_dict)
    fk = calculate_fk(angles_dict, leg_lengths, coxa_pos_dict)
    error = torch.mean((fk - p_3d)**2)
    return error

def rel_coxa_error(p_3d, true_coxa, scale = 0.3):
    ''' Coxa position error'''
    p_3d = torch.reshape(p_3d, (3, 6, 5, -1))
    norm = torch.distributions.normal.Normal(0, scale)
    e = torch.sqrt(torch.sum((p_3d[:, :, 0, :] - true_coxa)**2, axis = 1))
    n_e = torch.exp(norm.log_prob(e**2))
    n_zero = torch.exp(norm.log_prob(0))
    error = torch.mean((1 - n_e/n_zero)*e)
    return error

def rel_coxa_error_new(coxa, true_coxa, scale = 0.3):
    ''' Coxa position error (model parameters optimization)'''
    norm = torch.distributions.normal.Normal(0, scale)
    e = torch.sqrt(torch.sum((coxa - true_coxa)**2, axis = 1)+1e-10)
    n_e = torch.exp(norm.log_prob(e**2))
    n_zero = torch.exp(norm.log_prob(0))
    error = torch.mean((1 - n_e/n_zero)*e)
    return error

def leg_symmetry_error(p_3d):
    ''' Symmetry error'''
    fp = torch.reshape(p_3d, (3, 6, 5, -1))
    fp = torch.mean(fp, axis=3)
    leg_lengths = torch.norm(fp[:, :, 1:] - fp[:, :, :-1], dim = 0)
    error = torch.mean((leg_lengths[:3, :] - leg_lengths[3:, :])**2)
    return error

def leg_symmetry_error_new(legs):
    ''' Symmetry error (model parameters optimization)'''
    error = torch.mean((legs[:3, :] - legs[3:, :])**2)
    return error

def anatomy_error_1(p_3d):
    ''' Enforces leg planes'''
    p_3d = torch.reshape(p_3d, (3, 6, 5, -1))
    a = p_3d[:, :, 0, :] - p_3d[:, :, 1, :]
    b = p_3d[:, :, 1, :] - p_3d[:, :, 2, :]
    n = torch.cross(a, b, dim=0)
    n = n/(torch.norm(n, dim=0)+1e-10)
    c = p_3d[:, :, 3:, :] - p_3d[:, :, 0:1, :]
    dist = torch.einsum('ijk,ijlk->jlk', n, c)
    error = torch.sqrt(torch.sum(dist**2) + 1e-10)/dist.shape[-1]
    return error


def anatomy_error_2(rand, delta):
    ''' Limits rotation around Oz'''
    p_3d = torch.reshape(rand, (3, 6, 5, -1))
    meds = (p_3d[:, :3, 0, :] + p_3d[:, 3:, 0, :]) / 2
    a = torch.mean(meds[:, 2, :] - meds[:, 0, :], dim=-1)
    n = -torch.cross(a, torch.tensor([0., 1., 0.]))
    n = torch.stack([n[0], torch.tensor(0.), n[2]])
    n = n / torch.norm(n)
    femur = p_3d[:, :, 2, :] - p_3d[:, :, 0, :]
    femur = torch.stack([femur[0, :, :], torch.zeros(femur.shape[1:]), femur[2, :, :]])
    up = torch.einsum('k,kil->il', n, femur)
    down = torch.norm(femur, dim=0)
    cos = torch.true_divide(up, down)
    angles = torch.acos(cos) * 180 / np.pi
    angles[angles > 90] = angles[angles > 90] - 180
    cross = torch.cross(femur, einops.repeat(n, 'm -> m k n', k=6, n=femur.shape[-1]), dim=0)[1, :, :]
    angles = -torch.sign(cross) * angles
    angles = torch.cat([angles[:3, :], -angles[3:, :]], dim=0)

    func_front = get_func([[40, 80]], delta)
    func_middle = get_func([[-20, 20]], delta)
    func_hind = get_func([[-75, -40]], delta)

    res = torch.zeros(angles.shape)
    res[[0, 3], :] = func_front(angles[[0, 3], :])
    res[[1, 4], :] = func_middle(angles[[1, 4], :])
    res[[2, 5], :] = func_hind(angles[[2, 5], :])

    error = torch.sum(res ** 2)
    return error


def anatomy_error_3(rand, delta=20):
    ''' Angle limits within planes'''
    p_3d = torch.reshape(rand, (3, 6, 5, -1))
    legs = p_3d[:, :, 1:-1, :] - p_3d[:, :, :-2, :]
    legs = torch.cat([einops.repeat(torch.tensor([0, 0., -1]), 'm -> m k l n', k=6, l=1, n=legs.shape[-1]), legs],
                     dim=2)
    norms = torch.norm(legs, dim=0)
    down = norms[:, :-1, :] * norms[:, 1:, :]
    up = torch.einsum('kilm,kilm->ilm', legs[:, :, :-1, :], legs[:, :, 1:, :])
    cos = torch.true_divide(up, down)
    angles = torch.acos(cos) * 180 / np.pi

    func_front_tc = get_func([[-40, 10]], delta)
    func_front_cf = get_func([[85, 170]], delta)
    func_front_ft = get_func([[35, 140]], delta)
    func_front = lambda x: torch.stack([func_front_tc(x[:, 0, :]),
                                        func_front_cf(x[:, 1, :]),
                                        func_front_ft(x[:, 2, :])], dim=1)
    func_mid_tc = get_func([[15, 25]], delta)
    func_mid_cf = get_func([[90, 120]], delta)
    func_mid_ft = get_func([[90, 120]], delta)
    func_middle = lambda x: torch.stack([func_mid_tc(x[:, 0, :]),
                                         func_mid_cf(x[:, 1, :]),
                                         func_mid_ft(x[:, 2, :])], dim=1)
    func_hind_tc = get_func([[40, 60]], delta)
    func_hind_cf = get_func([[30, 110]], delta)
    func_hind_ft = get_func([[50, 145]], delta)
    func_hind = lambda x: torch.stack([func_hind_tc(x[:, 0, :]),
                                       func_hind_cf(x[:, 1, :]),
                                       func_hind_ft(x[:, 2, :])], dim=1)

    res = torch.zeros((6, 3, angles.shape[-1]))
    res[[0, 3], :, :] = func_front(angles[[0, 3]])
    res[[1, 4], :, :] = func_middle(angles[[1, 4]])
    res[[2, 5], :, :] = func_hind(angles[[2, 5]])

    error = torch.mean(res ** 2)
    return error


def anatomy_error_3_new(angles, delta=20):
    ''' Angle limits within planes (model parameters optimization)'''
    func_front_tc = get_func([[-40, 10]], delta)
    func_front_cf = get_func([[85, 170]], delta)
    func_front_ft = get_func([[35, 140]], delta)
    func_front = lambda x: torch.stack([func_front_tc(x[:, 0, :]),
                                        func_front_cf(x[:, 1, :]),
                                        func_front_ft(x[:, 2, :])], dim=1)
    func_mid_tc = get_func([[15, 25]], delta)
    func_mid_cf = get_func([[90, 120]], delta)
    func_mid_ft = get_func([[90, 120]], delta)
    func_middle = lambda x: torch.stack([func_mid_tc(x[:, 0, :]),
                                         func_mid_cf(x[:, 1, :]),
                                         func_mid_ft(x[:, 2, :])], dim=1)
    func_hind_tc = get_func([[40, 60]], delta)
    func_hind_cf = get_func([[30, 110]], delta)
    func_hind_ft = get_func([[50, 145]], delta)
    func_hind = lambda x: torch.stack([func_hind_tc(x[:, 0, :]),
                                       func_hind_cf(x[:, 1, :]),
                                       func_hind_ft(x[:, 2, :])], dim=1)

    res = torch.zeros((6, 3, angles.shape[-1]))
    res[[0, 3], :, :] = func_front(angles[[0, 3]])
    res[[1, 4], :, :] = func_middle(angles[[1, 4]])
    res[[2, 5], :, :] = func_hind(angles[[2, 5]])

    error = torch.mean(res ** 2)
    return error