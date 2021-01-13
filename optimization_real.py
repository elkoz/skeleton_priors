import torch
from losses import *
from utils import *
import pickle

with open('pose_result_fix__data_paper_180918_MDN_CsCh_Fly1_002_SG1_behData_images.pkl', 'rb') as f:
    file = pickle.load(f)

def optimize_real(random=10, a=1.5, legs=False, b=2.1, smooth=False, c=1, fk=False, d=3, coxa=False,
                  g=1.5, symmetry=False, anatomy_1=False, h=1.1, anatomy_2=False, f=1, anatomy_3=False, k=2,
                  repr_func=relaxed_repr_error_real, scale=0.1, delta=30, optimize_all=True,
                  optimizer=torch.optim.Adam, lr=0.01,
                  coxa_scale=0.3, epochs=500, change_epoch=300, fk_init=True):
    result = []
    center_r, focus_r, skew_r, tvec_r, angles_r, coef_r = pars_from_dict(file)

    rand, p = initialize(file)

    p_3d = torch.reshape(rand, (3, 6, 5, -1))
    true_coxa = torch.mean(p_3d[:, :, 0, :], axis=-1)
    true_coxa = torch.reshape(true_coxa, (3, 6, 1))

    if fk_init:
        pars = pars_from_points(rand)
        rand = construct(*pars)

    p_3d = torch.reshape(rand, (3, 6, 5, -1))
    true_coxa = torch.mean(p_3d[:, :, 0, :], axis=-1)
    true_coxa = torch.reshape(true_coxa, (3, 6, 1))

    rand.requires_grad = True

    losses_repr = []
    losses_legs = []
    losses_total = []
    losses_symm = []
    losses_anatomy1 = []
    losses_anatomy2 = []
    losses_anatomy3 = []
    res = []

    opt = optimizer([rand, center_r, focus_r, skew_r, tvec_r, angles_r], lr=lr)

    for e in range(epochs):
        loss = repr_func(p, rand, center_r, focus_r, skew_r, angles_r, tvec_r, coef_r, random)

        losses_repr.append(repr_error_real(p, rand, center_r, focus_r, skew_r, angles_r, tvec_r, coef_r, random))
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
        if anatomy_3:
            loss += (10. ** k) * anatomy_error_3(rand)

        losses_legs.append(leg_length_error(rand))
        losses_total.append(loss.data)
        losses_symm.append(leg_symmetry_error(rand))
        losses_anatomy1.append(anatomy_error_1(rand))
        losses_anatomy2.append(anatomy_error_2(rand, delta))
        losses_anatomy3.append(anatomy_error_3(rand, delta))
        if e in [24, 25]:
            res.append(rand.detach().clone())
        #         print(e)
        #         print(anatomy_error_2(rand, delta))
        #         res.append(rand.detach().clone())
        opt.zero_grad()
        loss.backward()
        opt.step()

    pars = [center_r, focus_r, skew_r, tvec_r, angles_r, coef_r]

    result.append((rand, losses_repr, losses_legs, losses_total, losses_anatomy3, losses_symm, losses_anatomy1,
                   losses_anatomy2, pars, res))
    return result
