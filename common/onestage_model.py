# Copyright (c) 2018-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
import torch
import torch.nn as nn
import numpy as np
from common.Block_lib import *
from common.embedding import Embedding


class TemporalBlock(nn.Module):
    """
    Reference 3D pose estimation model with temporal convolutions.
    This implementation can be used for all use-cases.
    """

    def __init__(self, num_joints_in, in_features, num_joints_out,
                 filter_widths, causal=False, dropout=0.2, channels=1024, latten_features=256, dense=False,
                 is_train=True, Optimize1f=True):
        super().__init__()

        self.is_train = is_train
        self.augment = False
        self.Optimize1f = Optimize1f
        self.num_joints_in = num_joints_in
        self.in_features = in_features
        self.num_joints_out = num_joints_out
        self.filter_widths = filter_widths

        self.drop = nn.Dropout(dropout)
        self.relu = nn.LeakyReLU(0.2, inplace=True)

        self.pad = [filter_widths[0] // 2]
        self.expand_bn = nn.BatchNorm1d(channels, momentum=0.1)
        # self.shrink = nn.Conv1d(channels, num_joints_out * 3, 1)
        self.shrink = nn.Conv1d(channels, latten_features, 1)

        if self.Optimize1f == False:
            self.expand_conv = nn.Conv1d(num_joints_in * in_features, channels, filter_widths[0], bias=False)
        else:
            self.expand_conv = nn.Conv1d(num_joints_in * in_features, channels, filter_widths[0],
                                         stride=filter_widths[0], bias=False)

        layers_conv = []
        layers_bn = []

        self.causal_shift = [(filter_widths[0]) // 2 if causal else 0]
        next_dilation = filter_widths[0]
        for i in range(1, len(filter_widths)):
            self.pad.append((filter_widths[i] - 1) * next_dilation // 2)
            self.causal_shift.append((filter_widths[i] // 2 * next_dilation) if causal else 0)

            if self.Optimize1f == False:
                layers_conv.append(nn.Conv1d(channels, channels,
                                             filter_widths[i] if not dense else (2 * self.pad[-1] + 1),
                                             dilation=next_dilation if not dense else 1,
                                             bias=False))
            else:
                layers_conv.append(nn.Conv1d(channels, channels, filter_widths[i], stride=filter_widths[i], bias=False))
            layers_bn.append(nn.BatchNorm1d(channels, momentum=0.1))
            layers_conv.append(nn.Conv1d(channels, channels, 1, dilation=1, bias=False))
            layers_bn.append(nn.BatchNorm1d(channels, momentum=0.1))

            next_dilation *= filter_widths[i]

        self.layers_conv = nn.ModuleList(layers_conv)
        self.layers_bn = nn.ModuleList(layers_bn)

    def set_bn_momentum(self, momentum):
        self.expand_bn.momentum = momentum
        for bn in self.layers_bn:
            bn.momentum = momentum

    def set_training_status(self, is_train):
        self.is_train = is_train

    def set_augment(self, augment):
        self.augment = augment

    def receptive_field(self):
        """
        Return the total receptive field of this model as # of frames.
        """
        frames = 0
        for f in self.pad:
            frames += f
        return 1 + 2 * frames

    def forward(self, x):
        x = self.drop(self.relu(self.expand_bn(self.expand_conv(x))))

        for i in range(len(self.pad) - 1):
            pad = self.pad[i + 1]
            shift = self.causal_shift[i + 1]
            if self.Optimize1f == False:
                res = x[:, :, pad + shift: x.shape[2] - pad + shift]
            else:
                res = x[:, :, self.causal_shift[i + 1] + self.filter_widths[i + 1] // 2:: self.filter_widths[i + 1]]

            x = self.drop(self.relu(self.layers_bn[2 * i](self.layers_conv[2 * i](x))))
            x = res + self.drop(self.relu(self.layers_bn[2 * i + 1](self.layers_conv[2 * i + 1](x))))

        x = self.shrink(x)
        x = x.permute(0, 2, 1)

        x_sz = x.shape
        x = x.reshape(x_sz[0] * x_sz[1], x_sz[2]).unsqueeze(1)

        return x





class TemporalBlock_angle(nn.Module):
    """
    Reference 3D pose estimation model with temporal convolutions.
    This implementation can be used for all use-cases.
    """

    def __init__(self, num_joints_in, in_features, num_joints_out,
                 filter_widths, causal=False, dropout=0.2, channels=1024, latten_features=256, dense=False,
                 is_train=True, Optimize1f=True):
        super().__init__()

        self.is_train = is_train
        self.augment = False
        self.Optimize1f = Optimize1f
        self.num_joints_in = num_joints_in
        self.in_features = in_features
        self.num_joints_out = num_joints_out
        self.filter_widths = filter_widths

        self.drop = nn.Dropout(dropout)
        self.relu = nn.LeakyReLU(0.2, inplace=True)

        self.pad = [filter_widths[0] // 2]
        self.expand_bn = nn.BatchNorm1d(channels, momentum=0.1)
        # self.shrink = nn.Conv1d(channels, num_joints_out * 3, 1)
        self.shrink = nn.Conv1d(channels, latten_features, 1)

        if self.Optimize1f == False:
            self.expand_conv = nn.Conv1d(num_joints_in, channels, filter_widths[0], bias=False)
        else:
            self.expand_conv = nn.Conv1d(num_joints_in, channels, filter_widths[0],
                                         stride=filter_widths[0], bias=False)

        layers_conv = []
        layers_bn = []

        self.causal_shift = [(filter_widths[0]) // 2 if causal else 0]
        next_dilation = filter_widths[0]
        for i in range(1, len(filter_widths)):
            self.pad.append((filter_widths[i] - 1) * next_dilation // 2)
            self.causal_shift.append((filter_widths[i] // 2 * next_dilation) if causal else 0)

            if self.Optimize1f == False:
                layers_conv.append(nn.Conv1d(channels, channels,
                                             filter_widths[i] if not dense else (2 * self.pad[-1] + 1),
                                             dilation=next_dilation if not dense else 1,
                                             bias=False))
            else:
                layers_conv.append(nn.Conv1d(channels, channels, filter_widths[i], stride=filter_widths[i], bias=False))
            layers_bn.append(nn.BatchNorm1d(channels, momentum=0.1))
            layers_conv.append(nn.Conv1d(channels, channels, 1, dilation=1, bias=False))
            layers_bn.append(nn.BatchNorm1d(channels, momentum=0.1))

            next_dilation *= filter_widths[i]

        self.layers_conv = nn.ModuleList(layers_conv)
        self.layers_bn = nn.ModuleList(layers_bn)

    def set_bn_momentum(self, momentum):
        self.expand_bn.momentum = momentum
        for bn in self.layers_bn:
            bn.momentum = momentum

    def set_training_status(self, is_train):
        self.is_train = is_train

    def set_augment(self, augment):
        self.augment = augment

    def receptive_field(self):
        """
        Return the total receptive field of this model as # of frames.
        """
        frames = 0
        for f in self.pad:
            frames += f
        return 1 + 2 * frames

    def forward(self, x):
        x = self.drop(self.relu(self.expand_bn(self.expand_conv(x))))

        for i in range(len(self.pad) - 1):
            pad = self.pad[i + 1]
            shift = self.causal_shift[i + 1]
            if self.Optimize1f == False:
                res = x[:, :, pad + shift: x.shape[2] - pad + shift]
            else:
                res = x[:, :, self.causal_shift[i + 1] + self.filter_widths[i + 1] // 2:: self.filter_widths[i + 1]]

            x = self.drop(self.relu(self.layers_bn[2 * i](self.layers_conv[2 * i](x))))
            x = res + self.drop(self.relu(self.layers_bn[2 * i + 1](self.layers_conv[2 * i + 1](x))))

        x = self.shrink(x)
        x = x.permute(0, 2, 1)

        x_sz = x.shape
        x = x.reshape(x_sz[0] * x_sz[1], x_sz[2]).unsqueeze(1)

        return x




class Linear(nn.Module):
    def __init__(self, linear_size, p_dropout=0.25):
        super(Linear, self).__init__()
        self.l_size = linear_size

        self.relu = nn.LeakyReLU(0.2, inplace=True)
        self.dropout = nn.Dropout(p_dropout)

        self.w1 = nn.Linear(self.l_size, self.l_size)
        self.batch_norm1 = nn.BatchNorm1d(self.l_size)

        self.w2 = nn.Linear(self.l_size, self.l_size)
        self.batch_norm2 = nn.BatchNorm1d(self.l_size)

    def forward(self, x):
        y = self.w1(x)
        y = self.batch_norm1(y)
        y = self.relu(y)
        y = self.dropout(y)

        y = self.w2(y)
        y = self.batch_norm2(y)
        y = self.relu(y)
        y = self.dropout(y)

        out = x + y

        return out


class FCBlock(nn.Module):

    def __init__(self, channel_in, channel_out, linear_size, block_num):
        super(FCBlock, self).__init__()

        self.linear_size = linear_size
        self.block_num = block_num
        self.layers = []
        self.channel_in = channel_in
        self.stage_num = 3
        self.p_dropout = 0.25
        self.fc_1 = nn.Linear(self.channel_in, self.linear_size)
        self.bn_1 = nn.BatchNorm1d(self.linear_size)
        for i in range(block_num):
            self.layers.append(Linear(self.linear_size, self.p_dropout))
        self.fc_2 = nn.Linear(self.linear_size, channel_out)

        self.layers = nn.ModuleList(self.layers)
        self.relu = nn.LeakyReLU(0.2, inplace=True)
        self.dropout = nn.Dropout(self.p_dropout)

    def forward(self, x):

        x = self.fc_1(x)
        x = self.bn_1(x)
        x = self.relu(x)
        x = self.dropout(x)
        for i in range(self.block_num):
            x = self.layers[i](x)
        x = self.fc_2(x)

        return x






####EHFusion
class nddr_layer(nn.Module):
    def __init__(self, in_channels, out_channels, task_num, init_weights=[0.9, 0.1], init_method='constant'):
        super(nddr_layer, self).__init__()
        self.task_num = task_num
        assert task_num>=2, 'Task Num Must >=2'


        self.Conv_Task_List = nn.ModuleList([])
        for i in range(self.task_num):
            task_basic = nn.Sequential(nn.Conv1d(in_channels, out_channels, 1),
                                       nn.BatchNorm1d(out_channels),
                                       nn.ReLU(True))
            self.Conv_Task_List.append(task_basic)
    def forward(self, Net_F):
        mix_features = self.Conv_Task_List[0](Net_F).squeeze(2)
        for i in range(1, self.task_num):
            tmp = self.Conv_Task_List[i](Net_F)
            new_tmp = tmp.squeeze(2)
            mix_features = torch.cat((mix_features, new_tmp), dim=1)
            # Net_Res.append(tmp)

        return mix_features








class One_Stage_Model(nn.Module):
    """
    Reference 3D pose estimation model with temporal convolutions.
    This implementation can be used for all use-cases.
    """

    def __init__(self, num_joints_in, in_features, num_joints_out,
                 filter_widths, causal=False, dropout=0.2, latten_features=256,
                 channels=1024, dense=False, is_train=True, Optimize1f=True, stage=1):
        super(One_Stage_Model, self).__init__()
        """
        Initialize this model.

        Arguments:
        num_joints_in -- number of input joints (e.g. 17 for Human3.6M)
        in_features -- number of input features for each joint (typically 2 for 2D input)
        num_joints_out -- number of output joints (can be different than input)
        filter_widths -- list of convolution widths, which also determines the # of blocks and receptive field
        causal -- use causal convolutions instead of symmetric convolutions (for real-time applications)
        dropout -- dropout probability
        channels -- number of convolution channels
        dense -- use regular dense convolutions instead of dilated convolutions (ablation experiment)
        is_train -- if the model runs in training mode or not
        Optimize1f=True -- using 1 frame optimization or not
        stage -- current stage when using the multi-stage optimization method
        """
        self.augment = False
        self.is_train = is_train
        self.num_joints_in = num_joints_in
        self.num_joints_out = num_joints_out
        self.in_features = in_features
        self.latten_features = latten_features
        self.stage = stage

        self.LocalLayer_Torso = TemporalBlock(5 * 3, in_features, num_joints_out, filter_widths, causal, dropout,
                                              channels, self.latten_features, dense, is_train, Optimize1f)
        self.LocalLayer_LArm = TemporalBlock(3 * 3, in_features, num_joints_out, filter_widths, causal, dropout,
                                             channels, self.latten_features, dense, is_train, Optimize1f)
        self.LocalLayer_RArm = TemporalBlock(3 * 3, in_features, num_joints_out, filter_widths, causal, dropout,
                                             channels, self.latten_features, dense, is_train, Optimize1f)
        self.LocalLayer_LLeg = TemporalBlock(3 * 3, in_features, num_joints_out, filter_widths, causal, dropout,
                                             channels, self.latten_features, dense, is_train, Optimize1f)
        self.LocalLayer_RLeg = TemporalBlock(3 * 3, in_features, num_joints_out, filter_widths, causal, dropout,
                                             channels, self.latten_features, dense, is_train, Optimize1f)




        self.pad = (self.receptive_field() - 1) // 2

        self.GlobalInfo = FCBlock(num_joints_in * 2, self.latten_features, 1024, 2)





        #####EHFusion
        self.NDDR_Fusion = nddr_layer(self.latten_features*4, int(self.latten_features/4), 4, init_weights=[0.9, 0.1], init_method='constant')
        self.FuseBlocks = FCBlock(self.latten_features * 4, self.latten_features, 1024, 1)

  





        embedd_dim = 64

        channels_angle = 256

        self.angle_torso = TemporalBlock_angle(5, in_features, num_joints_out, filter_widths, causal, dropout,
                                              channels_angle, embedd_dim, dense, is_train, Optimize1f)

        self.angle_larm = TemporalBlock_angle(3, in_features, num_joints_out, filter_widths, causal, dropout,
                                              channels_angle, embedd_dim, dense, is_train, Optimize1f)

        self.angle_rarm = TemporalBlock_angle(3, in_features, num_joints_out, filter_widths, causal, dropout,
                                              channels_angle, embedd_dim, dense, is_train, Optimize1f)

        self.angle_lleg = TemporalBlock_angle(3, in_features, num_joints_out, filter_widths, causal, dropout,
                                              channels_angle, embedd_dim, dense, is_train, Optimize1f)

        self.angle_rleg = TemporalBlock_angle(3, in_features, num_joints_out, filter_widths, causal, dropout,
                                              channels_angle, embedd_dim, dense, is_train, Optimize1f)


        self.extrinsic_dim = 4

        self.cam_embedd_dim = 64

        self.embedder = Embedding(in_channels=self.extrinsic_dim, out_channels=self.cam_embedd_dim)



        self.out_features_dim_S1 = self.latten_features * 2

        self.out_features_dim_S2 = self.latten_features * 3


        self.out_features_dim_S1 += embedd_dim

        self.out_features_dim_S2 += embedd_dim


        self.out_features_dim_S1 += self.cam_embedd_dim

        self.out_features_dim_S2 += self.cam_embedd_dim



        self.Integration_Torso_S1 = FCBlock(self.out_features_dim_S1, 5 * 3, 1024, 1)
        self.Integration_LArm_S1 = FCBlock(self.out_features_dim_S1, 3 * 3, 1024, 1)
        self.Integration_RArm_S1 = FCBlock(self.out_features_dim_S1, 3 * 3, 1024, 1)
        self.Integration_LLeg_S1 = FCBlock(self.out_features_dim_S1, 3 * 3, 1024, 1)
        self.Integration_RLeg_S1 = FCBlock(self.out_features_dim_S1, 3 * 3, 1024, 1)



        self.Integration_Torso_S2 = FCBlock(self.out_features_dim_S2, 5 * 3, 1024, 1)
        self.Integration_LArm_S2 = FCBlock(self.out_features_dim_S2, 3 * 3, 1024, 1)
        self.Integration_RArm_S2 = FCBlock(self.out_features_dim_S2, 3 * 3, 1024, 1)
        self.Integration_LLeg_S2 = FCBlock(self.out_features_dim_S2, 3 * 3, 1024, 1)
        self.Integration_RLeg_S2 = FCBlock(self.out_features_dim_S2, 3 * 3, 1024, 1)

        
    def set_bn_momentum(self, momentum):
        self.LocalLayer_Torso.set_bn_momentum(momentum)
        self.LocalLayer_LArm.set_bn_momentum(momentum)
        self.LocalLayer_RArm.set_bn_momentum(momentum)
        self.LocalLayer_LLeg.set_bn_momentum(momentum)
        self.LocalLayer_RLeg.set_bn_momentum(momentum)

    def set_training_status(self, is_train):
        self.is_train = is_train
        self.LocalLayer_Torso.set_training_status(is_train)
        self.LocalLayer_LArm.set_training_status(is_train)
        self.LocalLayer_RArm.set_training_status(is_train)
        self.LocalLayer_LLeg.set_training_status(is_train)
        self.LocalLayer_RLeg.set_training_status(is_train)

    def set_augment(self, augment):
        self.augment = augment
        self.LocalLayer_Torso.set_augment(augment)
        self.LocalLayer_LArm.set_augment(augment)
        self.LocalLayer_RArm.set_augment(augment)
        self.LocalLayer_LLeg.set_augment(augment)
        self.LocalLayer_RLeg.set_augment(augment)

    def receptive_field(self):
        """
        Return the total receptive field of this model as # of frames.
        """
        return self.LocalLayer_Torso.receptive_field()

    def forward(self, x, param):
        assert len(x.shape) == 4
        assert x.shape[-2] == self.num_joints_in
        assert x.shape[-1] == self.in_features

        pad = (self.receptive_field() - 1) // 2
        in_current = x[:, x.shape[1] // 2:x.shape[1] // 2 + 1]

        in_current = in_current.reshape(in_current.shape[0] * in_current.shape[1], -1)

        x_sz = x.shape

        x_ori = x.clone()


        x = x.view(x.shape[0], x.shape[1], -1)
        x = x.permute(0, 2, 1)

        sz = x.shape

        # Positional information encoding
        diff = x - x[:, 0:2, :].repeat(1, sz[1] // 2, 1)

        # Temporal information encoding
        diff_t = x - x[:, :, x.shape[2] // 2:x.shape[2] // 2 + 1].expand(sz[0], sz[1], sz[2])




        x_ori_relative = x_ori - x_ori[:, :, 0:1, :].repeat(1, 1, sz[1] // 2, 1)

        in_current_relative_root = x_ori_relative[:, x.shape[2] // 2:x.shape[2] // 2 + 1, :, :].expand(x_ori_relative.shape[0], x_ori_relative.shape[1], x_ori_relative.shape[2], x_ori_relative.shape[3])

        diff_cos = torch.cosine_similarity(x_ori_relative, in_current_relative_root, dim=3)


        diff_cos = torch.clamp(diff_cos, min=-1.0, max=1.0)

        diff_angle = torch.acos(diff_cos)

        diff_E_angle = torch.exp(diff_angle - np.pi)

        diff_E_angle = diff_E_angle.permute(0, 2, 1)

      



    


        # Grouping（分成躯干，左手，右手，左腿，右腿）
        in_Torso = torch.cat(
            (x[:, 0:2, :], x[:, 14:22, :], diff[:, 0:2, :], diff[:, 14:22, :], diff_t[:, 0:2, :], diff_t[:, 14:22, :]),
            dim=1)
        in_LArm = torch.cat((x[:, 28:34, :], diff[:, 28:34, :], diff_t[:, 28:34, :]), dim=1)
        in_RArm = torch.cat((x[:, 22:28, :], diff[:, 22:28, :], diff_t[:, 22:28, :]), dim=1)
        in_LLeg = torch.cat((x[:, 2:8, :], diff[:, 2:8, :], diff_t[:, 2:8, :]), dim=1)
        in_RLeg = torch.cat((x[:, 8:14, :], diff[:, 8:14, :], diff_t[:, 8:14, :]), dim=1)




        # Global Feature Encoder
        x_global = self.GlobalInfo(in_current)

        # Local Feature Encoder
        xTorso = self.LocalLayer_Torso(in_Torso)
        xLArm = self.LocalLayer_LArm(in_LArm)
        xRArm = self.LocalLayer_RArm(in_RArm)
        xLLeg = self.LocalLayer_LLeg(in_LLeg)
        xRLeg = self.LocalLayer_RLeg(in_RLeg)

        tmp = torch.cat((xTorso, xLArm, xRArm, xLLeg, xRLeg), dim=1)


        extrinsic_embedding = self.embedder(param)


        #Torso
        angle_torso_embedding = self.angle_torso(torch.cat((diff_E_angle[:, 0:1, :], diff_E_angle[:, 7:11, :]), dim=1))

        angle_torso_embedding = angle_torso_embedding.squeeze(dim=1)

        #Left arm
        angle_larm_embedding = self.angle_larm(diff_E_angle[:, 14:17, :])

        angle_larm_embedding = angle_larm_embedding.squeeze(dim=1)

        #Right arm
        angle_rarm_embedding = self.angle_rarm(diff_E_angle[:, 11:14, :])

        angle_rarm_embedding = angle_rarm_embedding.squeeze(dim=1)


        #Left leg
        angle_lleg_embedding = self.angle_lleg(diff_E_angle[:, 1:4, :])

        angle_lleg_embedding = angle_lleg_embedding.squeeze(dim=1)


        #Right leg
        angle_rleg_embedding = self.angle_rleg(diff_E_angle[:, 4:7, :])

        angle_rleg_embedding = angle_rleg_embedding.squeeze(dim=1)



        xTorso_S1 = torch.cat((tmp[:, 0], x_global, angle_torso_embedding, extrinsic_embedding), dim=1)
        xLArm_S1 = torch.cat((tmp[:, 1], x_global, angle_larm_embedding, extrinsic_embedding), dim=1)
        xRArm_S1 = torch.cat((tmp[:, 2], x_global, angle_rarm_embedding, extrinsic_embedding), dim=1)
        xLLeg_S1 = torch.cat((tmp[:, 3], x_global, angle_lleg_embedding, extrinsic_embedding), dim=1)
        xRLeg_S1 = torch.cat((tmp[:, 4], x_global, angle_rleg_embedding, extrinsic_embedding), dim=1)


        mix_features = torch.zeros(tmp.shape[0], 5, self.latten_features).cuda()

        

        xTorso_S2 = torch.cat((tmp[:, 0], mix_features[:, 0].detach(), x_global, angle_torso_embedding, extrinsic_embedding), dim=1)
        xLArm_S2 = torch.cat((tmp[:, 1], mix_features[:, 1].detach(), x_global, angle_larm_embedding, extrinsic_embedding), dim=1)
        xRArm_S2 = torch.cat((tmp[:, 2], mix_features[:, 2].detach(), x_global, angle_rarm_embedding, extrinsic_embedding), dim=1)
        xLLeg_S2 = torch.cat((tmp[:, 3], mix_features[:, 3].detach(), x_global, angle_lleg_embedding, extrinsic_embedding), dim=1)
        xRLeg_S2 = torch.cat((tmp[:, 4], mix_features[:, 4].detach(), x_global, angle_rleg_embedding, extrinsic_embedding), dim=1)


        # Decoder branch 1
        xTorso_S1 = self.Integration_Torso_S1(xTorso_S1)
        xLArm_S1 = self.Integration_LArm_S1(xLArm_S1)
        xRArm_S1 = self.Integration_RArm_S1(xRArm_S1)
        xLLeg_S1 = self.Integration_LLeg_S1(xLLeg_S1)
        xRLeg_S1 = self.Integration_RLeg_S1(xRLeg_S1)


        # Decoder branch 2
        xTorso_S2 = self.Integration_Torso_S2(xTorso_S2)
        xLArm_S2 = self.Integration_LArm_S2(xLArm_S2)
        xRArm_S2 = self.Integration_RArm_S2(xRArm_S2)
        xLLeg_S2 = self.Integration_LLeg_S2(xLLeg_S2)
        xRLeg_S2 = self.Integration_RLeg_S2(xRLeg_S2)





        #branch 1
        xTorso_S1 = xTorso_S1.view(xTorso_S1.size(0), 5, 3)
        xLArm_S1 = xLArm_S1.view(xLArm_S1.size(0), 3, 3)
        xRArm_S1 = xRArm_S1.view(xRArm_S1.size(0), 3, 3)
        xLLeg_S1 = xLLeg_S1.view(xLLeg_S1.size(0), 3, 3)
        xRLeg_S1 = xRLeg_S1.view(xRLeg_S1.size(0), 3, 3)



        #branch 2
        xTorso_S2 = xTorso_S2.view(xTorso_S2.size(0), 5, 3)
        xLArm_S2 = xLArm_S2.view(xLArm_S2.size(0), 3, 3)
        xRArm_S2 = xRArm_S2.view(xRArm_S2.size(0), 3, 3)
        xLLeg_S2 = xLLeg_S2.view(xLLeg_S2.size(0), 3, 3)
        xRLeg_S2 = xRLeg_S2.view(xRLeg_S2.size(0), 3, 3)



        #branch 1 non-fused pose
        x = torch.cat((xTorso_S1[:, 0:1], xLLeg_S1, xRLeg_S1, xTorso_S1[:, 1:5], xRArm_S1, xLArm_S1), dim=1)
        x = x.view(x_sz[0], x_sz[1] - 2 * pad, 17, 3)


        #branch 2 fused pose
        x_mixed = torch.cat((xTorso_S2[:, 0:1], xLLeg_S2, xRLeg_S2, xTorso_S2[:, 1:5], xRArm_S2, xLArm_S2), dim=1)
        x_mixed = x_mixed.view(x_sz[0], x_sz[1] - 2 * pad, 17, 3)


        return x, x_mixed
