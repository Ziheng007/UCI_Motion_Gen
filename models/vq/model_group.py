import random
import torch
import torch.nn as nn
from models.vq.encdec import Encoder, Decoder
from models.vq.residual_vq import ResidualVQ
from utils.humanml_utils import UPPER_JOINT_Y_MASK,HML_LEFT_ARM_MASK,HML_RIGHT_ARM_MASK,HML_LEFT_LEG_MASK,HML_RIGHT_LEG_MASK,OVER_LAP_LOWER_MASK,OVER_LAP_UPPER_MASK
import numpy as np
class RVQVAE_Group(nn.Module):
    def __init__(self,
                 args,
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
                 moment=True):

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
        self.encoder_left_arm = Encoder(arm_dim, output_emb_width//4, down_t, stride_t, width, depth,
                               dilation_growth_rate, activation=activation, norm=norm)
        self.encoder_right_arm = Encoder(arm_dim, output_emb_width//4, down_t, stride_t, width, depth,
                                dilation_growth_rate, activation=activation, norm=norm)
        self.encoder_left_leg = Encoder(leg_dim, output_emb_width//4, down_t, stride_t, width, depth,
                                dilation_growth_rate, activation=activation, norm=norm)
        self.encoder_right_leg = Encoder(leg_dim, output_emb_width//4, down_t, stride_t, width, depth,
                                dilation_growth_rate, activation=activation, norm=norm)
        
        # self.encoder_spine = Encoder(input_width, output_emb_width, down_t, stride_t, width, depth,
        #                 dilation_growth_rate, activation=activation, norm=norm)
        self.decoder = Decoder(input_width, output_emb_width, down_t, stride_t, width, depth,
                                 dilation_growth_rate, activation=activation, norm=norm)
        self.encoder = nn.ModuleList([self.encoder_left_arm,self.encoder_right_arm,self.encoder_left_leg,self.encoder_right_leg])
        self.body_mask =[HML_LEFT_ARM_MASK,HML_RIGHT_ARM_MASK,HML_LEFT_LEG_MASK,HML_RIGHT_LEG_MASK]
        self.upper_map = []
        self.lower_map = []
        self.update_mapper()
        # self.decoder = Decoder(input_width, output_emb_width, down_t, stride_t, width, depth,
        #                        dilation_growth_rate, activation=activation, norm=norm)
        rvqvae_config = {
            'num_quantizers': args.num_quantizers,
            'shared_codebook': args.shared_codebook,
            'quantize_dropout_prob': args.quantize_dropout_prob,
            'quantize_dropout_cutoff_index': 0,
            'nb_code': nb_code,
            'code_dim':code_dim//4, 
            'args': args,
        }
        self.quantizer = nn.ModuleList([ResidualVQ(**rvqvae_config) for _ in range(4)])
        #self.quantizer = [ResidualVQ(**rvqvae_config) for _ in range(4)]

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
        N, T, _ = x.shape
        x = self.shift_upper_down(x)
        x_in = self.preprocess(x)
        # print(x_encoder.shape)
        code_idx = []
        all_codes = []
        for encoder,mask,quantizer in zip(self.encoder,self.body_mask,self.quantizer):
            x_encoder = encoder(x_in[:,mask,:])#[bs,code_dim,16]
            # quantization
            code_idx_, all_codes_ = quantizer.quantize(x_encoder, return_latent=True)
            code_idx.append(code_idx_)
            all_codes.append(all_codes_)
        #TODO
        code_idx = torch.stack(code_idx,dim=1)
        all_codes = torch.stack(all_codes,dim=1)
        # print(f'code_idx shape {code_idx.shape}')
        # print(f'all_codes shape {all_codes.shape}')
        # print(code_idx.shape)
        # code_idx = code_idx.view(N, -1)
        # (N, T, Q)
        # print()
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
        x_in = self.preprocess(x)
        # Encode
        x_encode = []
        commit_loss = 0.
        perplexity = 0.
        for encoder,mask in zip(self.encoder,self.body_mask):
            x_encode_ = encoder(x_in[:,mask,:])#[bs,code_dim,16]
            x_encode.append(x_encode_)
        x_quantize = []
        for x,quantizer in zip(x_encode,self.quantizer):
            x_quantize_, code_idx, commit_loss_, perplexity_ = quantizer(x, sample_codebook_temp=0.5)
            commit_loss += commit_loss_
            perplexity += perplexity_
            x_quantize.append(x_quantize_)
        x_out = torch.cat(x_quantize,dim=1)
        x_out = self.decoder(x_out)
        x_out = self.shift_upper_up(x_out)
        # x_out = self.postprocess(x_decoder)
        return x_out, commit_loss, perplexity
    #TODO
    def forward_decoder(self, x):
        x_out = []
        for i, quantizer in enumerate(self.quantizer):
            x_d = quantizer.get_codes_from_indices(x[:, i])#[b,n,q]
            x_d = x_d.sum(dim=0).permute(0, 2, 1)
            x_out.append(x_d)
        x_out = torch.cat(x_out,dim=1)
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