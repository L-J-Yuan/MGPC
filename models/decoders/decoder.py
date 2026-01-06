import torch
import torch.nn as nn
import numpy as np



class MLP_Res(nn.Module):
    def __init__(self, in_dim=128, hidden_dim=None, out_dim=128):
        super(MLP_Res, self).__init__()
        if hidden_dim is None:
            hidden_dim = in_dim
        self.conv_1 = nn.Conv1d(in_dim, hidden_dim, 1)
        self.conv_2 = nn.Conv1d(hidden_dim, out_dim, 1)
        self.conv_shortcut = nn.Conv1d(in_dim, out_dim, 1) if in_dim != out_dim else nn.Identity()

    def forward(self, x):
        
        shortcut = self.conv_shortcut(x)
        out = self.conv_2(torch.relu(self.conv_1(x))) + shortcut
        return out


class MLP_CONV(nn.Module):
    def __init__(self, in_channel, layer_dims, bn=None):
        super(MLP_CONV, self).__init__()
        layers = []
        last_channel = in_channel
        for out_channel in layer_dims[:-1]:
            layers.append(nn.Conv1d(last_channel, out_channel, 1))
            if bn:
                layers.append(nn.BatchNorm1d(out_channel))
            layers.append(nn.ReLU())
            last_channel = out_channel
        layers.append(nn.Conv1d(last_channel, layer_dims[-1], 1))
        self.mlp = nn.Sequential(*layers)

    def forward(self, inputs):
        return self.mlp(inputs)




class Upsampling_unit(nn.Module):
   
    def __init__(self, up_ratio=2):
        super(Upsampling_unit, self).__init__()
        self.mlp = MLP_Res(in_dim=256, hidden_dim=128, out_dim=128)
        self.deconv_branch = nn.ConvTranspose1d(128, 128, up_ratio, up_ratio, bias=False) 
        self.duplicated_branch = nn.Upsample(scale_factor=up_ratio)

    def forward(self, feature, points):
        deconved_feat = self.deconv_branch(feature)      # b, 128(f), 2*n
        duplicated_feat = self.duplicated_branch(feature)        # b, 128(f), 2*n
        up_feat = self.mlp(torch.cat([deconved_feat, duplicated_feat], 1))    # b, 128(f), 2*n
        duplicated_point = self.duplicated_branch(points)       # b, 3, 2*n
        return up_feat, duplicated_point



class Upsampling(nn.Module):
   
    def __init__(self, in_channel=384, up_ratio=2):
        super(Upsampling, self).__init__()
        self.mlp = MLP_CONV(in_channel, [128, 128])
        self.up_unit = Upsampling_unit(up_ratio=up_ratio)  
        self.regressor = MLP_CONV(128, [64, 3])

    def forward(self, points, feature):
        # points: b, 3, n
        feature = self.mlp(feature)     # b, 128(f), n
        up_feat, duplicated_point = self.up_unit(feature, points)    # b, 128(f), 4*n    b, 3, 4*n
        offest = self.regressor(up_feat)        # b, 3(f), 4*n
        up_point = duplicated_point + offest        # b, 3, 4*n

        return up_point, up_feat


class Generator(torch.nn.Module):
    
    def __init__(self, up_ratio):
        super(Generator, self).__init__()
        up_rate = int(np.sqrt(up_ratio))
        self.upsampling_unit_1 = Upsampling(in_channel=384, up_ratio = up_rate)
        self.upsampling_unit_2 = Upsampling(in_channel=128, up_ratio = up_rate)


    def forward(self, point_cloud, rec_feat):
        p1_pred, up_feat = self.upsampling_unit_1(point_cloud, rec_feat)      # b, 3, 4*n      b, 128, 4*n
        p2_pred, up_feat = self.upsampling_unit_2(p1_pred, up_feat)      # b, 3, 16*n      b, 128, 16*n

        p2_pred = p2_pred.permute(0, 2, 1).contiguous()
        p1_pred = p1_pred.permute(0, 2, 1).contiguous()
        
        return p2_pred, p1_pred



class Coarse_Head(nn.Module):
    def __init__(self, input_dim=384):
        super().__init__()
        self.coarse_pred = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.GELU(),
            nn.Linear(128, 3)
        )

    def forward(self, feature):
        pred_low = self.coarse_pred(feature)
        return pred_low



class Progressive_Generator(nn.Module):
    def __init__(self):
        super().__init__()
        self.head = Coarse_Head()
        self.generator = Generator(16)

    def forward(self, rec_feat):
        
        pc_512 = self.head(rec_feat).permute(0, 2, 1).contiguous()      # b 3(f) 512(n)
        rec_feat = rec_feat.permute(0, 2, 1).contiguous()   # b, 384(f), 512(n)

        pc_8192, pc_2048 = self.generator(pc_512, rec_feat)

        return pc_8192, pc_2048, pc_512.permute(0, 2, 1).contiguous()


class Progressive_Generator_2048(nn.Module):
    def __init__(self):
        super().__init__()
        self.head = Coarse_Head()
        self.generator = Generator(4)

    def forward(self, rec_feat):
        
        pc_coarse = self.head(rec_feat).permute(0, 2, 1).contiguous()      # b 3(f) 512(n)
        rec_feat = rec_feat.permute(0, 2, 1).contiguous()   # b, 384(f), 512(n)

        pc_2048, pc_1024 = self.generator(pc_coarse, rec_feat)

        return pc_2048, pc_1024, pc_coarse.permute(0, 2, 1).contiguous()
