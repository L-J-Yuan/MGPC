import torch
import torch.nn as nn

from .encoders.image_encoder import DINOv2Tokenizer
from .encoders.text_encoder import FrozenImageCLIP
from .encoders.point_encoder import FeatureExtractor
from .decoders.decoder import Progressive_Generator, Progressive_Generator_2048
from .transformer.transformer import Transformer
from pointnet2_ops.pointnet2_utils import furthest_point_sample, gather_operation
from utils.metrics import HyperCD




class MGPC(nn.Module):

    def __init__(self, args=None):
        super().__init__()
        self.args = args
        self.point_encoder = FeatureExtractor()
        self.rgb_encoder = DINOv2Tokenizer()
        self.text_encoder = FrozenImageCLIP()
        self.backbone = Transformer(num_layers=8)
        self.decoder = Progressive_Generator()

        
    def encode(self, partial, rgb, text):

        bs = partial.shape[0]
        device = partial.device
        point_tokens = self.point_encoder(partial)      # b, 512, 384
        rgb_tokens = self.rgb_encoder(rgb)              # b, 257(16*16+1), 768
        text_tokens = self.text_encoder(bs, device, texts=text)           # b, 768

        return point_tokens, rgb_tokens, text_tokens
    
    def predict(self, partial, rgb, text):
        
        point_tokens, rgb_tokens, text_tokens = self.encode(partial, rgb, text)
        point_tokens = point_tokens.permute(0, 2, 1).contiguous()   # b, 384(f), 512(n)
        text_tokens = text_tokens.unsqueeze(1)
        multimodal_condition = torch.cat([rgb_tokens, text_tokens], dim=1)     # b, 258(n), 768(f)
        feature = self.backbone(point_tokens, multimodal_condition)               # b, 512, 384(f)
        pred_refine, pc_2048, pc_512 = self.decoder(feature)

        
        return pred_refine, pc_2048, pc_512

    def get_loss(self, x, gt, rgb, text):
        
        pred_refine, pc_2048, pc_512 = self.predict(x, rgb, text)
        
        # FPS
        gt_2048 = gather_operation(gt.permute(0, 2, 1).contiguous(), furthest_point_sample(gt, 2048)).permute(0, 2, 1).contiguous()
        gt_512 = gather_operation(gt.permute(0, 2, 1).contiguous(), furthest_point_sample(gt, 512)).permute(0, 2, 1).contiguous()
        
        # loss
        CD_loss_2048 = HyperCD(gt_2048, pc_2048)
        CD_loss_512 = HyperCD(gt_512, pc_512)
        CD_loss_refine = HyperCD(gt, pred_refine)
        loss = CD_loss_refine + CD_loss_2048 + CD_loss_512

        return loss



class MGPC_2048(nn.Module):

    def __init__(self, args=None):
        super().__init__()
        self.args = args
        self.point_encoder = FeatureExtractor()
        self.rgb_encoder = DINOv2Tokenizer()
        self.text_encoder = FrozenImageCLIP()
        self.backbone = Transformer(num_layers=8)
        self.decoder = Progressive_Generator_2048()

        
    def encode(self, partial, rgb, text):

        bs = partial.shape[0]
        device = partial.device
        point_tokens = self.point_encoder(partial)      # b, 512, 384
        rgb_tokens = self.rgb_encoder(rgb)              # b, 257(16*16+1), 768
        text_tokens = self.text_encoder(bs, device, texts=text)           # b, 768

        return point_tokens, rgb_tokens, text_tokens
    
    def predict(self, partial, rgb, text):
        
        point_tokens, rgb_tokens, text_tokens = self.encode(partial, rgb, text)
        point_tokens = point_tokens.permute(0, 2, 1).contiguous()   # b, 384(f), 512(n)
        text_tokens = text_tokens.unsqueeze(1)
        multimodal_condition = torch.cat([rgb_tokens, text_tokens], dim=1)     # b, 258(n), 768(f)
        feature = self.backbone(point_tokens, multimodal_condition)               # b, 512, 384(f)
        pred_2048, pc_1024, pc_512 = self.decoder(feature)

        
        return pred_2048, pc_1024, pc_512

    def get_loss(self, x, gt, rgb, text):
        
        pred_2048, pc_1024, pc_512, _, _, _ = self.predict(x, rgb, text)
        
        # FPS
        gt_2048 = gather_operation(gt.permute(0, 2, 1).contiguous(), furthest_point_sample(gt, 2048)).permute(0, 2, 1).contiguous()
        gt_1024 = gather_operation(gt.permute(0, 2, 1).contiguous(), furthest_point_sample(gt, 1024)).permute(0, 2, 1).contiguous()
        gt_512 = gather_operation(gt.permute(0, 2, 1).contiguous(), furthest_point_sample(gt, 512)).permute(0, 2, 1).contiguous()

        # loss
        CD_loss_1024 = HyperCD(gt_1024, pc_1024)
        CD_loss_512 = HyperCD(gt_512, pc_512)
        CD_loss_refine = HyperCD(gt_2048, pred_2048)
        loss = CD_loss_refine + CD_loss_1024 + CD_loss_512

        return loss

