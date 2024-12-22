import random
import torch
import torch.nn as nn
from models.vq.encdec import Encoder, Decoder
from models.vq.residual_vq import ResidualVQ
from utils.humanml_utils import UPPER_JOINT_Y_MASK,HML_LEFT_ARM_MASK,HML_RIGHT_ARM_MASK,HML_LEFT_LEG_MASK,HML_RIGHT_LEG_MASK,OVER_LAP_LOWER_MASK,OVER_LAP_UPPER_MASK
import numpy as np
class RVQVAE_Decode(nn.Module):
    def __init__(self,
                 args,
                 teacher_net,
                 input_width=263,
                 nb_code=1024,
                 code_dim=512,
                 output_emb_width=512,
                 down_t=3,
                 stride_t=2,
                 width=512,
                 depth=3,
                 dilation_growth_rate=3,
                 activation='relu',
                 norm=None,
                 mean=0.,
                 std=0.,
                 moment=True,):

        super().__init__()
        assert output_emb_width == code_dim
        self.code_dim = code_dim
        self.num_code = nb_code
        if moment:
            self.register_buffer('mean',torch.from_numpy(mean).to(torch.float32))
            self.register_buffer('std',torch.from_numpy(std).to(torch.float32))
            self.register_buffer('mean_upper', torch.tensor([0.1216, 0.2488, 0.2967, 0.5027, 0.4053, 0.4100, 0.5703, 0.4030, 0.4078, 0.1994, 0.1992, 0.0661, 0.0639], dtype=torch.float32))
            self.register_buffer('std_upper', torch.tensor([0.0164, 0.0412, 0.0523, 0.0864, 0.0695, 0.0703, 0.1108, 0.0853, 0.0847, 0.1289, 0.1291, 0.2463, 0.2484], dtype=torch.float32))
        # self.quant = args.quantizer
        if input_width == 263:
            leg_dim=59 #59+12
            arm_dim=108 #48+48
        #TODO 1 spine 2 input dim
        self.teacher_net = teacher_net
        self.decoder = Decoder(input_width, output_emb_width, down_t, stride_t, width, depth,
                               dilation_growth_rate, activation=activation, norm=norm)
        self.freeze_encoder_and_quantizer()

    def freeze_encoder_and_quantizer(self):
        for param in self.teacher_net.parameters():
            param.requires_grad = False
        print('Encoder and quantizer frozen')

    def preprocess(self, x):
        # (bs, T, Jx3) -> (bs, Jx3, T)
        x = x.permute(0, 2, 1).float()
        return x

    def postprocess(self, x):
        # (bs, Jx3, T) ->  (bs, T, Jx3)
        x = x.permute(0, 2, 1)
        return x
    #TODO
    def encode(self, x):
        code_idx, all_codes = self.teacher_net.encode(x)
        return code_idx, all_codes
    
    def normalize(self,data):
        return (data-self.mean)/self.std
    def denormalize(self,data):
        return data*self.std+self.mean
    def normalize_upper(self, data):
        return (data - self.mean_upper) / self.std_upper
    def denormalize_upper(self, data):
        return data * self.std_upper + self.mean_upper
    
    def shift_upper_down(self, data):
        data = data.clone()
        data = self.denormalize(data)
        shift_y = data[..., 3:4].clone()
        data[..., UPPER_JOINT_Y_MASK] -= shift_y
        _data = data.clone()
        data = self.normalize(data)
        data[..., UPPER_JOINT_Y_MASK] = self.normalize_upper(_data[..., UPPER_JOINT_Y_MASK])
        return data
    def shift_upper_up(self, data):
        _data = data.clone()
        data = self.denormalize(data)
        data[..., UPPER_JOINT_Y_MASK] = self.denormalize_upper(_data[..., UPPER_JOINT_Y_MASK])
        shift_y = data[..., 3:4].clone()
        data[..., UPPER_JOINT_Y_MASK] += shift_y
        data = self.normalize(data)
        return data
    
    def update_mapper(self):
        overlap_upper_mask = HML_RIGHT_ARM_MASK & HML_LEFT_ARM_MASK
        overlap_lower_mask = HML_RIGHT_LEG_MASK & HML_LEFT_LEG_MASK
        upper_indices = np.nonzero(overlap_upper_mask)[0]
        lower_indices = np.nonzero(overlap_lower_mask)[0]
        cnt=0
        for i,t in enumerate(HML_RIGHT_LEG_MASK):
            if i in lower_indices:
                self.lower_map.append(cnt)
                cnt+=1
            elif t:
                cnt+=1
        cnt=0
        for i,t in enumerate(HML_LEFT_ARM_MASK):
            if i in upper_indices:
                self.upper_map.append(cnt)
                cnt+=1
            elif t:
                cnt+=1

    def merge_upper_lower(self,x):
        bs,f = x[0].shape[:2]
        motion = torch.empty(x[0].shape[:2] + (263,)).to(x[0].device)
        # motion = torch.empty(bs,f,263).to(x.device)
        for mask,x_ in zip(self.body_mask,x): 
            motion[...,mask] = x_
        motion[...,OVER_LAP_LOWER_MASK]=(x[2][...,self.lower_map]+x[3][...,self.lower_map])/2
        motion[...,OVER_LAP_UPPER_MASK]=(x[0][...,self.upper_map]+x[1][...,self.upper_map])/2
        return motion
    
    def forward(self, x):
        x = self.shift_upper_down(x)
        x_out, commit_loss, perplexity = self.teacher_net.quantize(x)
        x_out = self.decoder(x_out)
        x_out = self.shift_upper_up(x_out)
        return x_out, commit_loss, perplexity

    def forward_decoder(self, x):
        x_out = self.teacher_net.dequantize(x)
        x_out = self.decoder(x_out)
        x_out = self.shift_upper_up(x_out)
        return x_out

class LengthEstimator(nn.Module):
    def __init__(self, input_size, output_size):
        super(LengthEstimator, self).__init__()
        nd = 512
        self.output = nn.Sequential(
            nn.Linear(input_size, nd),
            nn.LayerNorm(nd),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Dropout(0.2),
            nn.Linear(nd, nd // 2),
            nn.LayerNorm(nd // 2),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Dropout(0.2),
            nn.Linear(nd // 2, nd // 4),
            nn.LayerNorm(nd // 4),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Linear(nd // 4, output_size)
        )

        self.output.apply(self.__init_weights)

    def __init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    def forward(self, text_emb):
        return self.output(text_emb)