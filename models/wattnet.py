from .mlp import MLP
from .modules import *


class WATTNet(nn.Module):
    def __init__(self, series_len, in_dim, out_dim, w_dim=32, emb_dim=8, depth=4, dropout_prob=0.2, n_repeat=2):
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
                      with short sequences without running into situations where the dilated kernel becomes wider than the
                      sequence itself.
        """
        super().__init__()
        self.w_dim = w_dim
        self.emb_dim = emb_dim
        self.dilation_depth = depth
        self.n_layers = depth * n_repeat
        self.dilations = [2 ** i for i in range(1, depth + 1)] * n_repeat

        ltransf_dim = w_dim * emb_dim
        self.attblocks = nn.ModuleList([AttentionBlock(in_channels=w_dim,
                                                       key_size=ltransf_dim,
                                                       value_size=ltransf_dim)
                                        for _ in self.dilations])

        self.resblocks = nn.ModuleList([GatedBlock(dilation=d, w_dim=w_dim)
                                        for d in self.dilations])

        self.emb_conv = nn.Conv2d(1, emb_dim, kernel_size=1)
        self.dec_conv = nn.Conv2d(w_dim, w_dim, kernel_size=(1, emb_dim), groups=w_dim)

        # feature compression: when more memory/data is available, increasing w_dim can yield
        # better performance
        self.pre_mlp = MLP(in_dim, w_dim, out_softmax=False)

        # post fully-connected head not always necessary. When sequence length perfectly aligns
        # with the number of time points lost to high dilation, (i.e single latent output by
        # alternating TCN and attention modules) the single latent can be used directly
        self.post_mlp = MLP(w_dim, in_dim, [512], out_softmax=False, drop_probability=dropout_prob)
        self.output_fc = nn.Linear(series_len - sum(self.dilations), out_dim)

    def forward(self, x_in):
        """
        Args:
            x_in: 'N, C, H, W' where `N` is the batch dimension, `C` the one-hot
                  embedding dimension, `H` is the temporal dimension, `W` is the
                  second dimension of the timeseries (e.g timeseries for different FX pairs)
        Returns:
        """
        # x_in = self.preMLP(x_in.squeeze(1))
        x_in = self.pre_mlp(x_in)
        x_in = x_in.unsqueeze(1)

        if self.emb_dim > 1:
            x_in = self.emb_conv(x_in)

        # swap `W` dim to channel dimension for grouped convolutions
        # `N, W, H, C`
        x_in = x_in.transpose(1, 3)

        skip_connections = []
        for i in range(len(self.resblocks)):
            x_in = self.resblocks[i](x_in)
            x_att_list = []
            # slicing across `H`, temporal dimension
            for k in range(x_in.size(2)):
                # `C` embedding message passing using self-att
                x_att = self.attblocks[i](x_in[:, :, k, :])
                # `N, W, C` -> `N, W, 1, C`
                x_att = x_att.unsqueeze(2)
                x_att_list.append(x_att)
                # `N, W, 1, C` -> `N, W, H, C`
            x_in = torch.cat(x_att_list, dim=2)
        # `N, W, H, C` ->  `N, W, H, 1`
        if self.emb_dim > 1:
            x_in = self.dec_conv(x_in)
        # `N, W, H, 1` ->  `N, 1, H, W`
        x_out = x_in.transpose(1, 3)
        # `N, 1, H, W` ->  `N, H, W`
        x_out = x_out[:, 0, :, :]

        x_out = self.post_mlp(x_out)  # N, H, sensor_num
        x_out = self.output_fc(x_out.transpose(1, 2))
        x_out = x_out.transpose(1, 2)  # N, future_len, sensor_num
        return x_out

    @property
    def name(self):
        return type(self).__name__
