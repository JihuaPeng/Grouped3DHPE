# Copyright (c) 2018-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import torch
import numpy as np

def mpjpe(predicted, target):
    """
    Mean per-joint position error (i.e. mean Euclidean distance),
    often referred to as "Protocol #1" in many papers.
    """
    assert predicted.shape == target.shape
    return torch.mean(torch.norm(predicted - target, dim=len(target.shape)-1))


def mse(predicted, target):
    """
    Mean per-joint position error (i.e. mean Euclidean distance),
    often referred to as "Protocol #1" in many papers.
    """
    assert predicted.shape == target.shape
    return np.mean(np.linalg.norm(predicted - target, axis=len(target.shape)-1))

def L1_norm(predicted, target):
    """
    Mean per-joint position error (i.e. mean Euclidean distance),
    often referred to as "Protocol #1" in many papers.
    """
    assert predicted.shape == target.shape
    return torch.mean(torch.norm(predicted - target, dim=len(target.shape)-1, p=1))

def angle_loss(predicted, target):
    """
    Mean per-joint position error (i.e. mean Euclidean distance),
    often referred to as "Protocol #1" in many papers.
    """
    assert predicted.shape == target.shape
    return torch.mean(torch.cosine_similarity(predicted,target, dim=len(target.shape)-1))

def weighted_mpjpe(predicted, target, w):
    """
    Weighted mean per-joint position error (i.e. mean Euclidean distance)
    """
    assert predicted.shape == target.shape
    assert w.shape[0] == predicted.shape[0]
    return torch.mean(w * torch.norm(predicted - target, dim=len(target.shape)-1))

def p_mpjpe(predicted, target):
    """
    Pose error: MPJPE after rigid alignment (scale, rotation, and translation),
    often referred to as "Protocol #2" in many papers.
    """
    assert predicted.shape == target.shape
    
    muX = np.mean(target, axis=1, keepdims=True)
    muY = np.mean(predicted, axis=1, keepdims=True)
    
    X0 = target - muX
    Y0 = predicted - muY

    normX = np.sqrt(np.sum(X0**2, axis=(1, 2), keepdims=True))
    normY = np.sqrt(np.sum(Y0**2, axis=(1, 2), keepdims=True))
    
    X0 /= normX
    Y0 /= normY

    H = np.matmul(X0.transpose(0, 2, 1), Y0)
    U, s, Vt = np.linalg.svd(H)
    V = Vt.transpose(0, 2, 1)
    R = np.matmul(V, U.transpose(0, 2, 1))

    # Avoid improper rotations (reflections), i.e. rotations with det(R) = -1
    sign_detR = np.sign(np.expand_dims(np.linalg.det(R), axis=1))
    V[:, :, -1] *= sign_detR
    s[:, -1] *= sign_detR.flatten()
    R = np.matmul(V, U.transpose(0, 2, 1)) # Rotation

    tr = np.expand_dims(np.sum(s, axis=1, keepdims=True), axis=2)

    a = tr * normX / normY # Scale
    t = muX - a*np.matmul(muY, R) # Translation
    
    # Perform rigid transformation on the input
    predicted_aligned = a*np.matmul(predicted, R) + t
    
    # Return MPJPE
    return np.mean(np.linalg.norm(predicted_aligned - target, axis=len(target.shape)-1))
    
def n_mpjpe(predicted, target):
    """
    Normalized MPJPE (scale only), adapted from:
    https://github.com/hrhodin/UnsupervisedGeometryAwareRepresentationLearning/blob/master/losses/poses.py
    """
    assert predicted.shape == target.shape
    
    norm_predicted = torch.mean(torch.sum(predicted**2, dim=3, keepdim=True), dim=2, keepdim=True)
    norm_target = torch.mean(torch.sum(target*predicted, dim=3, keepdim=True), dim=2, keepdim=True)
    scale = norm_target / norm_predicted
    return mpjpe(scale * predicted, target)

def mean_velocity_error(predicted, target):
    """
    Mean per-joint velocity error (i.e. mean Euclidean distance of the 1st derivative)
    """
    assert predicted.shape == target.shape
    
    velocity_predicted = np.diff(predicted, axis=0)
    velocity_target = np.diff(target, axis=0)
    
    return np.mean(np.linalg.norm(velocity_predicted - velocity_target, axis=len(target.shape)-1))



def PoseAngle_loss(predicted, target):

    assert predicted.shape == target.shape

    #############predicted angle#############

    E_predicted_cos1 = torch.cosine_similarity(predicted, predicted, dim=len(target.shape) - 1)

    E_predicted_cos1 = torch.clamp(E_predicted_cos1, min=-1.0, max=1.0)

    E_predicted_angle1 = torch.arccos(E_predicted_cos1)

    E_predicted_angle1_exp_pi = torch.exp(E_predicted_angle1 - np.pi)

    E_predicted_angle1_exp_pi = E_predicted_angle1_exp_pi.repeat(1, 17, 1)



    P_predicted = predicted.unsqueeze(3).repeat(1, 1, 1, 17 ,1)

    P_t_predicted = P_predicted.permute(0, 1, 3, 2, 4)

    E_predicted_cos2 = torch.cosine_similarity(P_predicted, P_t_predicted, dim=len(target.shape))

    E_predicted_cos2 = torch.clamp(E_predicted_cos2, min=-1.0, max=1.0)

    E_predicted_angle2 = torch.arccos(E_predicted_cos2)

    E_predicted_angle2_exp_pi = torch.exp(E_predicted_angle2 - np.pi)

    E_predicted_angle2_exp_pi = E_predicted_angle2_exp_pi.squeeze(1)




    E_predicted_angle3_exp_pi = E_predicted_angle1_exp_pi.permute(0, 2, 1)



    # E_predicted_cos1 = E_predicted_cos1.repeat(1, 17, 1)

    # E_predicted_cos3 = E_predicted_cos1.permute(0, 2, 1)

    E_predicted_all = E_predicted_angle1_exp_pi - 2*E_predicted_angle2_exp_pi + E_predicted_angle3_exp_pi


    # E_predicted_cos = torch.cosine_similarity(predicted, predicted,dim=len(target.shape) - 1)
    # E_predicted_angle = torch.arccos(E_predicted_cos)
    # E_predicted = torch.exp(E_predicted_angle - np.pi)

    #############predicted angle#############




    #############target angle###############

    E_target_cos1 = torch.cosine_similarity(target, target, dim=len(target.shape) - 1)

    E_target_cos1 = torch.clamp(E_target_cos1, min=-1.0, max=1.0)

    E_target_angle1 = torch.arccos(E_target_cos1)

    E_target_angle1_exp_pi = torch.exp(E_target_angle1 - np.pi)

    E_target_angle1_exp_pi = E_target_angle1_exp_pi.repeat(1, 17, 1)




    P_target = target.unsqueeze(3).repeat(1, 1, 1, 17, 1)

    P_t_target = P_target.permute(0, 1, 3, 2, 4)

    E_target_cos2 = torch.cosine_similarity(P_target, P_t_target, dim=len(target.shape))

    ############E_target_cos2中有-1，-1求arccos是nan############

    E_target_cos2 = torch.clamp(E_target_cos2, min=-1.0, max=1.0)

    E_target_angle2 = torch.arccos(E_target_cos2)

    E_target_angle2_exp_pi = torch.exp(E_target_angle2 - np.pi)

    E_target_angle2_exp_pi = E_target_angle2_exp_pi.squeeze(1)



    E_target_angle3_exp_pi = E_target_angle1_exp_pi.permute(0, 2, 1)




    E_target_all = E_target_angle1_exp_pi - 2 * E_target_angle2_exp_pi + E_target_angle3_exp_pi

    # E_target_cos1 = torch.cosine_similarity(target, target, dim=len(target.shape) - 1)
    #
    # P_target = target.unsqueeze(3).repeat(1, 1, 1, 17, 1)
    #
    # P_t_target = P_target.permute(0, 1, 3, 2, 4)
    #
    # E_target_cos2 = torch.cosine_similarity(P_target, P_t_target, dim=len(target.shape))
    #
    # E_target_cos2 = E_target_cos2.squeeze(1)
    #
    # E_target_cos1 = E_target_cos1.repeat(1, 17, 1)
    #
    # E_target_cos3 = E_target_cos1.permute(0, 2, 1)
    #
    # E_target_all = E_target_cos1 - 2 * E_target_cos2 + E_target_cos3


    # E_target_cos = torch.cosine_similarity(target[:, :, 15], target[:, :, 16] ,dim=len(target.shape) - 2)
    # E_target_angle = torch.arccos(E_target_cos)
    # E_target = torch.exp(E_target_angle - np.pi)

    #############target angle###############

    # loss_angle = torch.mean(torch.abs(E_predicted_all - E_target_all))


    return torch.mean(torch.abs(E_predicted_all - E_target_all))


def per_joint_mpjpe(predicted, target):

    assert predicted.shape == target.shape
    residual_error = torch.mean(torch.norm(predicted - target, dim=len(target.shape)-1), dim=len(target.shape)-4)

    return residual_error[0]
