import torch.nn as nn
from ops.esa import ESA



class IDSG(nn.Module):
    def __init__(self, channel_num=64, bias = True, block_num=4,**kwargs):
        super(IDSG, self).__init__()

        ffn_bias    = kwargs.get("ffn_bias", False)
        window_size = kwargs.get("window_size", 0)
        pe          = kwargs.get("pe", False)

        block_script_name   = kwargs["block_script_name"]
        block_class_name    = kwargs["block_class_name"]

        script_name     = "ops." + block_script_name
        package         = __import__(script_name, fromlist=True)
        block_class     = getattr(package, block_class_name)
        self.residual_layer = block_class(channel_num,bias,ffn_bias=ffn_bias,window_size=window_size,with_pe=pe)
        self.res_end = nn.Conv2d(channel_num,channel_num,1,1,0,bias=bias)
        esa_channel     = max(channel_num // 4, 16)
        self.esa        = ESA(esa_channel, channel_num)
        
    def forward(self, x, a1, a2, a3, a4, a5, a6):
        out = self.residual_layer(x, a1, a2, a3, a4, a5, a6)
        out = self.res_end(out)
        out = out + x
        return self.esa(out)