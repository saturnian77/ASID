import torch.nn as nn
from ops.esa import ESA
from ops.IDSA import IDSA_Block1



class IDSG_A(nn.Module):
    def __init__(self, channel_num=64, bias = True, block_num=4,**kwargs):
        super(IDSG_A, self).__init__()

        ffn_bias    = kwargs.get("ffn_bias", False)
        window_size = kwargs.get("window_size", 0)
        pe          = kwargs.get("pe", False)

        group_list = []
        for _ in range(block_num):
            temp_res = IDSA_Block1(channel_num,bias,ffn_bias=ffn_bias,window_size=window_size,with_pe=pe)
            group_list.append(temp_res)
        self.res_end = nn.Conv2d(channel_num,channel_num,1,1,0,bias=bias)
        self.residual_layer = nn.Sequential(*group_list)
        esa_channel     = max(channel_num // 4, 16)
        self.esa        = ESA(esa_channel, channel_num)
        
    def forward(self, x):
        out, a1, a2, a3, a4, a5, a6 = self.residual_layer(x)
        out = self.res_end(out)
        out = out + x
        return self.esa(out), a1, a2, a3, a4, a5, a6