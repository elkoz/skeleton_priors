import torch
from utils import *
from losses import *
import numpy as np
#df3dPostProcessing: from Victor Lobato Rios
from df3dPostProcessing.df3dPostProcessing import df3dPostProcess
import pickle

num_cam = 6
init_angles = torch.tensor([-np.pi/2, 0, 0])
angles = get_circle()
center = torch.tensor([[480., 240.] for i in range(num_cam)])
focus = torch.tensor([[16041.0, 15971.7] for i in range(num_cam)])
skew = torch.zeros(num_cam)
tvec = torch.tensor([[0, 0, 100.] for i in range(num_cam)])
coef = torch.zeros(5, 6, 1)

leg_name = 'allExtraDOF_grooming'

experiment = './experiment/pose_result__home_nely_Desktop_DF3D_data_180921_aDN_CsCh_Fly6_003_SG1_behData_images_images.pkl'

df3d = df3dPostProcess(experiment)
align = df3d.align_3d_data()
leg_angles = df3d.calculate_leg_angles()

fly_points = get_3d(leg_angles, align, 0, 10)
fly_points = fly_points.reshape((3, -1))

with open('pose_result_fix__data_paper_180918_MDN_CsCh_Fly1_002_SG1_behData_images.pkl', 'rb') as f:
    file = pickle.load(f)

p = projection(fly_points, center, focus, skew, angles, init_angles, tvec, coef)

fp = torch.reshape(fly_points, (3, 6, 5, -1))
fp = fp[:, :, :, 0:1]
leg_lengths = torch.norm(fp[:, :, 1: :] - fp[:, :, :-1, :], dim = 0)

leg_names = ['RH_leg','RM_leg','RF_leg','LH_leg','LM_leg','LF_leg']


def optimize(random=10, a=1.9, legs=True, b=2.1, smooth=True, c=1, fk=False, d=4, coxa=False,
             g=1.5, symmetry=False, anatomy_1=False, h=1.1, anatomy_2=False, f=1, delta=15,
             repr_func=relaxed_repr_error, scale=0.1, optimize_all=True, optimizer=torch.optim.Adam,
             lr=0.01, coxa_scale=0.3, epochs=1300, change_epoch=800, seeds=[0]):
    result = []
    i = 2

    p_3d = torch.reshape(fly_points, (3, 6, 5, -1))
    true_coxa = torch.mean(p_3d[:, :, 0, :], axis=-1)
    true_coxa = torch.reshape(true_coxa, (3, 6, 1))

    for i in seeds:
        torch.manual_seed(i)
        p_rand = p.data + random * torch.normal(torch.zeros(p.shape), torch.ones(p.shape))
        if optimize_all:
            center_r = center.data + 2 * torch.normal(torch.zeros(center.shape), torch.ones(center.shape))
            center_r.requires_grad = True
            focus_r = focus.data + 5 * torch.normal(torch.zeros(focus.shape), torch.ones(focus.shape))
            focus_r.requires_grad = True
            skew_r = skew.data + 0.01 * torch.normal(torch.zeros(skew.shape), torch.ones(skew.shape))
            skew_r.requires_grad = True
            tvec_r = tvec.data + 0.7 * torch.normal(torch.zeros(tvec.shape), torch.ones(tvec.shape))
            tvec_r.requires_grad = True
            angles_r = angles.data + 0.01 * torch.normal(torch.zeros(angles.shape), torch.ones(angles.shape))
            angles_r.requires_grad = True
            init_angles_r = init_angles.data + 0.01 * torch.normal(torch.zeros(init_angles.shape),
                                                                   torch.ones(init_angles.shape))
            init_angles_r.requires_grad = True

        else:
            center_r = center
            focus_r = focus
            skew_r = skew
            tvec_r = tvec
            angles_r = angles
            init_angles_r = init_angles

        rand = triangulate(p_rand, init_angles_r, angles_r, focus_r, center_r, skew_r, tvec_r)
        rand.requires_grad = True

        losses_repr = []
        losses_proc = []
        losses_legs = []
        res = []
        losses_fk = []

        if optimize_all:
            opt = optimizer([rand, center_r, focus_r, skew_r, angles_r, init_angles_r, tvec_r], lr=lr)
        else:
            opt = optimizer([rand], lr=lr)

        for e in range(epochs):
            loss = repr_func(p_rand, rand, center_r, focus_r, skew_r, angles_r, init_angles_r, tvec_r, coef, random)
            if smooth:
                loss += (10. ** b) * smooth_error(rand)
            if fk and e > change_epoch:
                loss += (10. ** c) * fk_error(rand)
            if legs:
                loss += (10. ** a) * leg_length_error(rand)
            if symmetry:
                loss += (10. ** g) * leg_symmetry_error(rand)
            if coxa:
                loss += (10. ** d) * rel_coxa_error(rand, true_coxa=true_coxa, scale=coxa_scale)
            if anatomy_1:
                loss += (10. ** h) * anatomy_error_1(rand)
            if anatomy_2:
                loss += (10. ** f) * anatomy_error_2(rand, delta)

            losses_repr.append(
                repr_error(p, rand, center_r, focus_r, skew_r, angles_r, init_angles_r, tvec_r, coef, random))
            losses_legs.append(leg_length_error(rand))
            #             res.append(rand.clone().detach())
            if e % 50 == 0:
                res.append(rand.detach().clone())
            opt.zero_grad()
            loss.backward()
            opt.step()

        result.append((rand, losses_repr, losses_proc, losses_legs, res, losses_fk))
        print(i)
    return result


