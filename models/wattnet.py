from .mlp import MLP
from .modules import *
import torch

class WATTNet(nn.Module):
    def __init__(self, series_len, in_dim, out_dim, w_dim=16, emb_dim=16, depth=2, dropout_prob=0.2, n_repeat=1, feat_dim=0,
                 show_attn_alpha=False):
        """
        Args:
            w_dim: spatial compression dimension carried out by a 2-layer MLP.
                           When more memory/data is available, increasing w_dim can yield better performance
            emb_dim: embedding dimension of scalar values for each of the `w_dim` left after compression.
                     Higher embedding dimension increases accuracy of the spatial attention module at the cost
                     of increased memory requirement. BEWARE: w_dim * emb_dim > 1e4 can get *VERY* costly in terms
                     of GPU memory, especially with big batches.
            depth: number of temporal-spatial blocks. Dilation for temporal dilated convolution is doubled
                            each time.
            n_repeat: number of repeats of #`dilation_depth` of temporal-spatial layers. Useful to increase model depth
                      with short sequences without running into situations where the dilated kernel becomes wider than
                      the sequence itself.
            show_attn_alpha: whether to show the matrix `alpha` in the self-attention module
        """
        super().__init__()
        # self.w_dim = w_dim
        self.w_dim = in_dim
        self.emb_dim = emb_dim
        self.dilation_depth = depth
        self.n_layers = depth * n_repeat
        self.show_attention_alpha = show_attn_alpha
        self.feat_dim = feat_dim
        self.dilations = [2 ** i for i in range(1, depth + 1)] * n_repeat

        ltransf_dim = self.w_dim * emb_dim
        self.attention_blocks = nn.ModuleList([AttentionBlock(in_channels=self.w_dim,
                                                              key_size=ltransf_dim,
                                                              value_size=ltransf_dim)
                                               for _ in self.dilations])
        length = list()
        ResBlocks = list()
        total_len = series_len
        for d in self.dilations:
            length.append(total_len-d)
            ResBlocks.append(GatedBlock(dilation=d, w_dim=self.w_dim,seq_in=total_len,seq_out=total_len-d))
            total_len -= d

        self.resblocks = nn.ModuleList(ResBlocks)

        # self.emb_conv = nn.Conv2d(1+self.feat_dim, emb_dim, kernel_size=1)
        self.emb_conv = nn.Linear(1+self.feat_dim,emb_dim)
        # self.dec_conv = nn.Linear(emb_dim,1)
        self.dec_conv = nn.Conv2d(self.w_dim, self.w_dim, kernel_size=(1, emb_dim), groups=self.w_dim)

        # feature compression: when more memory/data is available, increasing w_dim can yield
        # better performance
        # self.pre_mlp = MLP(in_dim, self.w_dim, out_softmax=False)

        # post fully-connected head not always necessary. When sequence length perfectly aligns
        # with the number of time points lost to high dilation, (i.e single latent output by
        # alternating TCN and attention modules) the single latent can be used directly
        # self.post_mlp = MLP(self.w_dim, in_dim, [32], out_softmax=False, drop_probability=dropout_prob)
        self.output_fc = nn.Linear(series_len - sum(self.dilations), out_dim)

    def forward(self, x_in):
        """
        Args:
            x_in: 'N, H, W' where `N` is the batch dimension, `C` the one-hot
                  embedding dimension, `H` is the temporal dimension, `W` is the
                  second dimension of the timeseries (e.g timeseries for different FX pairs)
            x_in: B*T*(N+3)
        Returns:
        """
        # x_in = self.pre_mlp(x_in)
        feature = x_in[...,-3:]
        x_in = x_in[...,:-3]
        B,T,N = x_in.shape

        feature = feature.unsqueeze(2).repeat(1,1,N,1) # B*T*3->B*T*1*3->B*T*N*3
        x_in = x_in.unsqueeze(3)  # `N, H, W` -> `N, C, H, W`
        # x_in = torch.cat([x_in,feature[...,-1:]],dim=-1)  # B*T*N*4

        if self.emb_dim > 1:
            x_in = self.emb_conv(x_in.reshape(-1,x_in.shape[-1]))
            x_in = x_in.reshape(B,T,N,-1)   # B*T*N*F


        # swap `W` dim to channel dimension for grouped convolutions
        # `N, W, H, C`
        x_in = x_in.transpose(1, 2)     # B*N*T*F

        skip_connections = []
        for i in range(len(self.resblocks)):
            x_in = self.resblocks[i](x_in)
            x_att_list = []

            # slicing across `H`, temporal dimension
            temporal_len = x_in.size(2)
            for k in range(temporal_len):
                # `C` embedding message passing using self-att
                if k == temporal_len - 1 and self.show_attention_alpha:
                    x_att = self.attention_blocks[i](x_in[:, :, k, :], str(i + 1))
                else:
                    x_att = self.attention_blocks[i](x_in[:, :, k, :])
                # `N, W, C` -> `N, W, 1, C`
                x_att = x_att.unsqueeze(2)
                x_att_list.append(x_att)

            # `N, W, 1, C` -> `N, W, H, C`
            x_in = torch.cat(x_att_list, dim=2)

        # `N, W, H, C` ->  `N, W, H, 1`
        if self.emb_dim > 1:
            # x_in = torch.cat([x_in,feature[...,-x_in.shape[2]:,1:]],dim=-1)
            x_in = self.dec_conv(x_in)
        # `N, W, H, 1` ->  `N, 1, H, W`
        x_out = x_in.transpose(1, 3)
        # `N, 1, H, W` ->  `N, H, W`
        x_out = x_out[:, 0, :, :]

        # x_out = self.post_mlp(x_out)  # N, H, sensor_num
        x_out = self.output_fc(x_out.transpose(1, 2))
        x_out = x_out.transpose(1, 2)  # N, future_len, sensor_num
        # x_out = x_in
        # if self.emb_dim > 1:
        #     x_out = self.dec_conv(x_out.reshape(-1,self.emb_dim))
        #     x_out = x_out.reshape(B,N,-1)
        # x_out = self.output_fc(x_out.reshape(B*N,-1))
        # x_out = x_out.reshape(B,N,-1).transpose(1,2)

        return x_out

    @property
    def name(self):
        return type(self).__name__
