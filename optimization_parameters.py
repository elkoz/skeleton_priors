import torch
from utils import get_3d, get_circle, triangulate
from losses import *
from utils import *
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


def optimize_new(random=10, a=1.9, legs=True, b=2.1, smooth=True, c=1, fk=False, d=4, coxa_stable=False,
                 g=1.5, symmetry=False, anatomy_1=False, h=1.1, anatomy_2=False, f=1, delta=15,
                 repr_func=relaxed_repr_error, scale=0.1, optimize_all=True, optimizer=torch.optim.Adam,
                 lr=0.01, coxa_scale=0.3, epochs=1300, change_epoch=800, seeds=[0]):
    result = []

    for i in seeds:
        torch.manual_seed(i)
        angles = get_circle()
        p_rand = p.data + random * torch.normal(torch.zeros(p.shape), torch.ones(p.shape))
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

        losses_repr = []

        rand = triangulate(p_rand, init_angles_r, angles_r, focus_r, center_r, skew_r, tvec_r)
        coxa, joint_angles, legs = pars_from_points(rand)
        true_coxa = coxa.clone()

        coxa.requires_grad = True
        angles.requires_grad = True
        legs.requires_grad = True

        if optimize_all:
            opt = optimizer([coxa, joint_angles, legs, center_r, focus_r, skew_r, angles_r, init_angles_r, tvec_r],
                            lr=lr)
        else:
            opt = optimizer([rand], lr=lr)

        for e in range(epochs):
            loss = repr_func(p_rand, coxa, joint_angles, legs, center_r, focus_r, skew_r, angles_r, init_angles_r,
                             tvec_r, coef, random)
            if smooth:
                loss += (10. ** b) * smooth_error(coxa, joint_angles, legs)
            if symmetry:
                loss += (10. ** g) * leg_symmetry_error(legs)
            if coxa_stable:
                loss += (10. ** d) * rel_coxa_error(coxa, true_coxa=true_coxa, scale=coxa_scale)
            if anatomy_2:
                loss += (10. ** f) * anatomy_error_2(angles, delta)

            if e % 10 == 0:
                err = repr_error(p, coxa, joint_angles, legs, center_r, focus_r, skew_r, angles_r, init_angles_r,
                                 tvec_r, coef, random)
                losses_repr.append(err)
                print(f'e: {e}, loss: {loss}, err: {err}')
            opt.zero_grad()
            loss.backward()
            opt.step()

        result.append((construct(coxa, joint_angles, legs), losses_repr))
        print(i)
    return result
