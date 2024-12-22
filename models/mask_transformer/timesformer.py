import torch.nn as nn
import torch
from torch import einsum
from torch.nn import functional as F
from einops import rearrange, repeat
from timesformer_pytorch.rotary import apply_rot_emb, AxialRotaryEmbedding, RotaryEmbedding

def exists(val):
    return val is not None

class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.fn = fn
        self.norm = nn.LayerNorm(dim)

    def forward(self, x, *args, **kwargs):
        x = self.norm(x)
        return self.fn(x, *args, **kwargs)

# time token shift

def shift(t, amt):
    if amt is 0:
        return t
    return F.pad(t, (0, 0, 0, 0, amt, -amt))

class PreTokenShift(nn.Module):
    def __init__(self, frames, fn):
        super().__init__()
        self.frames = frames
        self.fn = fn

    def forward(self, x, *args, **kwargs):
        f, dim = self.frames, x.shape[-1]
        cls_x, x = x[:, :1], x[:, 1:]
        x = rearrange(x, 'b (f n) d -> b f n d', f = f)

        # shift along time frame before and after

        dim_chunk = (dim // 3)
        chunks = x.split(dim_chunk, dim = -1)
        chunks_to_shift, rest = chunks[:3], chunks[3:]
        shifted_chunks = tuple(map(lambda args: shift(*args), zip(chunks_to_shift, (-1, 0, 1))))
        x = torch.cat((*shifted_chunks, *rest), dim = -1)

        x = rearrange(x, 'b f n d -> b (f n) d')
        x = torch.cat((cls_x, x), dim = 1)
        return self.fn(x, *args, **kwargs)

# feedforward

class GEGLU(nn.Module):
    def forward(self, x):
        x, gates = x.chunk(2, dim = -1)
        return x * F.gelu(gates)

class FeedForward(nn.Module):
    def __init__(self, dim, mult = 4, dropout = 0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, dim * mult * 2),
            GEGLU(),
            nn.Dropout(dropout),
            nn.Linear(dim * mult, dim)
        )

    def forward(self, x):
        return self.net(x)
    
def attn(q, k, v, mask = None):
    sim = einsum('b i d, b j d -> b i j', q, k)
    #TODO
    if exists(mask):
        #max_neg_value = torch.tensor(float('-inf'))
        max_neg_value = -1e9
        sim.masked_fill_(~mask, max_neg_value)

    attn = sim.softmax(dim = -1)
    out = einsum('b i j, b j d -> b i d', attn, v)
    return out
    
class Time_Attention(nn.Module):
    def __init__(
        self,
        dim,
        dim_head = 64,
        heads = 8,
        dropout = 0.
    ):
        super().__init__()
        self.heads = heads
        self.scale = dim_head ** -0.5
        inner_dim = dim_head * heads

        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias = False)
        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x, einops_from, einops_to, mask = None, rot_emb = None, **einops_dims):
        h = self.heads
        q, k, v = self.to_qkv(x).chunk(3, dim = -1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> (b h) n d', h = h), (q, k, v))
        q = q * self.scale
        # rearrange across time or space
        q, k, v = map(lambda t: rearrange(t, f'{einops_from} -> {einops_to}', **einops_dims), (q, k, v))
        # add rotary embeddings, if applicable
        if exists(rot_emb):
            q, k = apply_rot_emb(q, k, rot_emb)
        # attention
        out = attn(q, k, v, mask = mask)
        # merge back time or space
        out = rearrange(out, f'{einops_to} -> {einops_from}', **einops_dims)
        # merge back the heads
        out = rearrange(out, '(b h) n d -> b n (h d)', h = h)
        # combine heads out
        return self.to_out(out)
        
class TimeSformer(nn.Module):
    def __init__(
        self,
        *,
        dim,
        num_frames,
        image_height = 5,
        image_width = 32,
        patch_height = 1,
        patch_width = 32,
        channels = 1,
        depth = 1,
        heads = 8,
        dim_head = 64,
        attn_dropout = 0.1,
        ff_dropout = 0.1,
        rotary_emb = True,
        shift_tokens = False
    ):
        super().__init__()
        assert image_height % patch_height == 0, 'Image height must be divisible by the patch height.'
        assert image_width % patch_width == 0, 'Image width must be divisible by the patch width.'

        num_patches = (image_height // patch_height) * (image_width // patch_width)
        num_positions = num_frames * num_patches
        patch_dim = channels * patch_height * patch_width

        self.heads = heads
        self.patch_height = patch_height
        self.patch_width = patch_width
        self.to_patch_embedding = nn.Linear(patch_dim, dim)
        self.cls_token = nn.Parameter(torch.randn(1, dim))

        self.use_rotary_emb = rotary_emb
        if rotary_emb:
            self.frame_rot_emb = RotaryEmbedding(dim_head)
            self.image_rot_emb = AxialRotaryEmbedding(dim_head)
        else:
            self.pos_emb = nn.Embedding(num_positions + 1, dim)

        self.layers = nn.ModuleList([])
        for _ in range(depth):
            ff = FeedForward(dim, dropout = ff_dropout)
            time_attn = Time_Attention(dim, dim_head = dim_head, heads = heads, dropout = attn_dropout)
            spatial_attn = Time_Attention(dim, dim_head = dim_head, heads = heads, dropout = attn_dropout)

            if shift_tokens:
                time_attn, spatial_attn, ff = map(lambda t: PreTokenShift(num_frames, t), (time_attn, spatial_attn, ff))

            time_attn, spatial_attn, ff = map(lambda t: PreNorm(dim, t), (time_attn, spatial_attn, ff))

            self.layers.append(nn.ModuleList([time_attn, spatial_attn, ff]))

    def forward(self, video, mask = None):
        b, f, _, h, w, *_, device, ph, pw = *video.shape, video.device, self.patch_height, self.patch_width
        assert h % ph == 0 and w % pw == 0, f'height {h} and width {w} of video must be divisible by the patch height {ph} and patch width {pw}'
        # calculate num patches in height and width dimension, and number of total patches (n)
        hp, wp = (h // ph), (w // pw)
        n = hp * wp
        # video to patch embeddings
        x = rearrange(video, 'b f c (hp ph) (wp pw) -> b (f hp wp) (ph pw c)', ph=ph, pw=pw)
        # tokens = self.to_patch_embedding(video)
        # positional embedding
        frame_pos_emb = None
        image_pos_emb = None
        if not self.use_rotary_emb:
            x += self.pos_emb(torch.arange(x.shape[1], device = device))
        else:
            frame_pos_emb = self.frame_rot_emb(f, device = device)
            # TODO delete
            image_pos_emb = self.image_rot_emb(hp, wp, device = device)
        # calculate masking for uneven number of frames
        frame_mask = None
        if exists(mask):
            # mask_with_cls = F.pad(mask, (1, 0), value = True)
            frame_mask = repeat(mask, 'b f -> (b h n) () f', n = n, h = self.heads)
        # time and space attention
        for (time_attn, spatial_attn, ff) in self.layers:
            x = time_attn(x, 'b (f n) d', '(b n) f d', n = n, mask = frame_mask, rot_emb = frame_pos_emb) + x
            x = spatial_attn(x, 'b (f n) d', '(b f) n d', f = f, rot_emb = image_pos_emb) + x
            x = ff(x) + x
        return x