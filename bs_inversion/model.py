import torch
from torch import nn
import numpy as np

class ResnetBlock(nn.Module):
    """Residual Block
    Args:
        in_channels (int): number of channels in input data
        out_channels (int): number of channels in output 
    """
    def __init__(self, in_channels, out_channels, kernel_size=3, one_d=False):
        super(ResnetBlock, self).__init__()
        self.build_conv_block(in_channels, out_channels, one_d, kernel_size=kernel_size)

    def build_conv_block(self, in_channels, out_channels, one_d, kernel_size=3):
        padding = (kernel_size -1)//2
        if not one_d:
            conv = nn.Conv2d
            norm = nn.BatchNorm2d
        else:
            conv = nn.Conv1d
            norm = nn.BatchNorm1d

        self.conv1 = nn.Sequential(
            conv(in_channels, out_channels, kernel_size=kernel_size, padding=padding),
            norm(out_channels),
            nn.ELU()
        )
        self.conv2 = nn.Sequential(
            conv(out_channels, out_channels, kernel_size=kernel_size, padding=padding),
            norm(out_channels),
        )
        if in_channels != out_channels:
            self.down = nn.Sequential(
                conv(in_channels, out_channels, kernel_size=1, bias=False),
                norm(out_channels)
            )
        else:
            self.down = None
        
        self.act = nn.ELU()

    def forward(self, x):
        """
        Args:
            x (Tensor): B x C x T
        """
        residual = x
        out = self.conv1(x)
        out = self.conv2(out)
        if self.down is not None:
            residual = self.down(residual)
        return self.act(out + residual)


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, padding=1, one_d=False):
        super(ConvBlock, self).__init__()
        if not one_d:
            self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=padding)
            self.bn = nn.BatchNorm2d(out_channels)
        else:
            self.conv = nn.Conv1d(in_channels, out_channels, kernel_size=kernel_size, padding=padding)
            self.bn = nn.BatchNorm1d(out_channels)

        self.act = nn.ELU()
    
    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.act(x)
        return x

class MaxOut2D(nn.Module):
    """
    Pytorch implementation of MaxOut on channels for an input that is C x H x W.
    Reshape input from N x C x H x W --> N x H*W x C --> perform MaxPool1D on dim 2, i.e. channels --> reshape back to
    N x C//maxout_kernel x H x W.
    """
    def __init__(self, max_out):
        super(MaxOut2D, self).__init__()
        self.max_out = max_out
        self.max_pool = nn.MaxPool1d(max_out)

    def forward(self, x):
        batch_size = x.shape[0]
        channels = x.shape[1]
        height = x.shape[2]
        width = x.shape[3]
        # Reshape input from N x C x H x W --> N x H*W x C
        x_reshape = torch.permute(x, (0, 2, 3, 1)).view(batch_size, height * width, channels)
        # Pool along channel dims
        x_pooled = self.max_pool(x_reshape)
        # Reshape back to N x C//maxout_kernel x H x W.
        return torch.permute(x_pooled, (0, 2, 1)).view(batch_size, channels // self.max_out, height, width).contiguous()


class MaxOut1D(nn.Module):
    """
    Pytorch implementation of MaxOut on channels for an input that is C x H x W.
    Reshape input from N x C x H x W --> N x H*W x C --> perform MaxPool1D on dim 2, i.e. channels --> reshape back to
    N x C//maxout_kernel x H x W.
    """
    def __init__(self, kernel_size=1, stride=2):
        super(MaxOut1D, self).__init__()
        self.max_out = kernel_size
        self.max_pool = nn.MaxPool1d(kernel_size=kernel_size, stride=stride)

    def forward(self, x):
        batch_size = x.shape[0]
        channels = x.shape[1]
        layer = x.shape[2]
        #print(f'channels={channels}')
        #print(f'self.max_out={self.max_out}')

        #print(x.shape)
        # Reshape input from B x C x L --> B x L x C
        x_reshape = torch.permute(x, (0, 2, 1))
        #print(x_reshape.shape)
        # Pool along channel dims
        x_pooled = self.max_pool(x_reshape)
        #print(x_pooled.shape)
        # Reshape back to B x C//maxout_kernel x L.
        return torch.permute(x_pooled, (0, 2, 1)).contiguous()

class MidLayer(nn.Module):
    """Applies 1D pooling/conv1d operator over input tensor.

    Args:
        in_channels (int): number of input channels
        out_channels (int): number of output channels
        residuals (int, optional): number of residual blocks. Default=0
    """
    def __init__(self, in_channels, out_channels, residuals=0, b_maxout=False):
        super(MidLayer, self).__init__()
        # TODO: try umsampling with bilinear interpolation 
        #self.upsample = nn.Upsample(scale_factor=2, mode='linear')
        #consider switching that with Pool1d
        self.b_maxout = b_maxout
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size=3, padding=1) 
        self.maxout = MaxOut1D(kernel_size=1, stride=2)
        torch.nn.init.xavier_uniform_(self.conv.weight)
        self.bn = nn.BatchNorm1d(out_channels)
        self.act = nn.ELU()     
        # self.pool = nn.MaxPool1d(kernel_size=2, stride=2, padding=0)
        
        if residuals != 0:
            # resnet blocks
            layers = []
            for _ in range(residuals):
                layers.append(
                    ResnetBlock(out_channels, out_channels, one_d=True)
                    )
            self.res_blocks = nn.Sequential(*layers)
        else:
            self.res_blocks = None


    def forward(self, x):
        """
        Args:
            x (Tensor): B x in_channels x T
        
        Returns:
            Tensor of shape (B, out_channels, T x 2)
        """
        # pool network
        B, C, T = x.shape
        # upsample
        # x = x.unsqueeze(dim=3)
        # x = F.upsample(x, size=(T*2, 1), mode='bilinear').squeeze(3)
        #x = self.upsample(x)
        # x = self.pad(x)
        
        if self.b_maxout == True:
            x = self.maxout(x)
        else:
            x = self.conv(x)
        x = self.bn(x)
        x = self.act(x)
        #x = self.pool(x)

        # pass through resnet blocks to improve internal representations
        # of data
        if self.res_blocks != None:
            x = self.res_blocks(x)
        return x
    
class HeadBS2(nn.Module):
    """HeadBS2 module

    Args:
        channels (list): list of #channels in each upsampling layer
        pre_residuals (int, optional): number of residual blocks before upsampling. Default: 64
        down_conv_channels (list): list of #channels in each down_conv blocks
        up_residuals (int, optional): number of residual blocks in each upsampling module. Default: 0
    """
    def __init__(self, input_len, channels,
          pre_residuals,
          pre_conv_channels,
          up_residuals,
          b_maxout,
          post_residuals,
          pow_2_channels,
          reduce_height
          ):
        super(HeadBS2, self).__init__()
        # Initialize learnable output factor
        self.f = torch.nn.Parameter(torch.ones(1))
        
        self.dropout = nn.Dropout(p=0.3)
        # Create pre_conv layer
        pre_convs = []
        bs_channels = 2
        c0 = pre_conv_channels[0]
        
        # add first pre_conv layer: bs_channels-->pre_conv_channels[0]
        # add residuals after layer
        pre_convs.append(ConvBlock(bs_channels, c0, kernel_size=3, padding=1))
        for _ in range(pre_residuals):
            pre_convs.append(ResnetBlock(c0, c0))
        
        # add additional pre_convs layer: pre_conv_channels[i]-->pre_conv_channels[i + 1]
        # add residuals after each layer
        # pre_conv_channels set by input parameter
        for i in range(len(pre_conv_channels) -1):
            in_c = pre_conv_channels[i]
            out_c = pre_conv_channels[i + 1]
            pre_convs.append(ResnetBlock(in_c, out_c))
            for _ in range(pre_residuals):
                pre_convs.append(ResnetBlock(out_c, out_c))
        self.pre_conv = nn.Sequential(*pre_convs)
        
        c1 = pre_conv_channels[-1]
        # Create pre middle layer - reduce height
        reduce_height_layers = []
        cnt, k, s = reduce_height
        for _ in range(cnt):
            reduce_height_layers.append(nn.Conv2d(in_channels=c1, out_channels=c1, 
                kernel_size=(k, 1), stride=(s, 1)))
        self.reduce_height = nn.Sequential(*reduce_height_layers)
        
        # Create middle layer - reduce channels
        mid_layers = []
        
        # set input channel
        if pow_2_channels:
            # create channels ranging from c1 to next smaller power of 2, keep until getting 8
            # additional layer for every channel
            get_closest_pow2_d = lambda x: np.power(2, int(np.log(x) / np.log(2)))
            c2 = get_closest_pow2_d(c1)#512, 256, 128, 64, 32, 16, 8
            mid_layers.append(ConvBlock(c1, c2, kernel_size=3, padding=1, one_d=True))
            #print(f'ConvBlock {ch0}, {ch1}')

            while c2/2. > 0:
                in_channels = c2
                out_channels = int(c2/2.)
                c2 = int(c2/2.)
                layer = MidLayer(in_channels, out_channels, residuals=up_residuals, b_maxout=b_maxout)
                mid_layers.append(layer)
                #print(f'mid_layers {in_channels}, {out_channels}')
                if int(c2) == 8:
                    break
        else:
            # create layer for c1-->c2
            c2 = 8
            mid_layers.append(ConvBlock(c1, c2, kernel_size=3, padding=1, one_d=True))
           
        self.mid = nn.Sequential(*mid_layers)
        
        # Create post layer - only residuals, count set by input parameter
        post_convs = []
        last_channels = int(c2)
        for i in range(post_residuals):
            post_convs.append(ResnetBlock(last_channels, last_channels, one_d=True, kernel_size=5))
        self.post_conv = nn.Sequential(*post_convs)

    def forward(self, x):
        """
        forward pass
        Args:
            x (Tensor): B x C x T # 100X100X2

        Returns:
            Tensor: B x C x (2^#channels * T) # 100X100X(2^#channels * 2)
        """
        #print(x.shape)
        x = x.unsqueeze(0) # reshape to [B x 2 x 100 x 100]
        #print(x.shape)

        x = self.pre_conv(x)
        #print(x.shape)
        ##x = self.dropout(x)
        # for BXCXHXW reduce dimension to BXCX1XW
        #s1, _, _, s4 = x.shape
        #x = x.reshape(s1, -1, s4)
        x = self.reduce_height(x)
        #print(x.shape)
        x = x.squeeze(2)
        #print(x.shape)

        x = self.mid(x)
        #print(x.shape)
        x *= self.f
        
        x2 = self.post_conv(x)
        #print(x2.shape)
        x2 *= self.f
        
        return x, x2# pre, post


class CNNBS2(nn.Module):
    """CNNBS2  - CNN for BS inversion

    Args:
        n_heads (int): Number of heads
        layer_channels (list): list of #channels of each layer
    """
    def __init__(self, input_len, n_heads, 
         channels,
         b_maxout,
         pre_conv_channels,
         pre_residuals, 
         up_residuals,
         post_residuals,
         pow_2_channels,
         reduce_height):
        # channels=[]
        # bs_channels = 2
        # ch1= int(input_len * pre_conv_channels[-1] * bs_channels)
        # channels.append(ch1)
        # ch2 = int(np.sqrt(np.power(2, np.floor(np.log(ch1)/np.log(2)))))
        # channels.append(ch2)
        # while ch2/2. > 0:
        #     ch2 /= 2.
        #     channels.append(int(ch2))
        #     if int(ch2) == 8:
        #        break
        # print(len(channels))
        super(CNNBS2, self).__init__()
        self.n_heads = n_heads
        self.linear = nn.Linear(8, 1)#channels[-1]
        if n_heads > 1:
            self.act_fn = nn.LeakyReLU()#nn.Softsign()
        else:
            self.act_fn = nn.Softsign()
            
        self.heads = nn.ModuleList([HeadBS2(input_len, channels, b_maxout=b_maxout,
                pre_conv_channels=pre_conv_channels, 
                pre_residuals=pre_residuals, up_residuals=up_residuals,
                post_residuals=post_residuals, pow_2_channels=pow_2_channels, reduce_height=reduce_height)
                                    for _ in range(n_heads)])

    def forward(self, x):

        # Pass over Heads in "parallel"
        if self.n_heads > 1:
            pre_list = []
            post_list = []
            #print('Pass over Heads in parallel- start')

            for head in self.heads:
                pre, post = head(x)

                pre_list.append(pre)
                post_list.append(post)
                #print(head.f)
            #print('Pass over Heads in parallel- end')
            # Sum Heads outputs
            pre = torch.sum(torch.stack(pre_list), dim=0)
            post = torch.sum(torch.stack(post_list), dim=0)
        else:
            # Pass over Head
            pre, post = self.heads[0](x)
            
        # Pre output
        rs0 = self.linear(pre.transpose(1, 2))
        rs0 = self.act_fn(rs0).squeeze(-1)
        
        # Post output - actually used
        rs1 = self.linear(post.transpose(1, 2))
        rs1 = self.act_fn(rs1).squeeze(-1)
        
        return rs0, rs1 

class HeadBS1(nn.Module):
    """HeadBS1 module - Original one with reshape

    Args:
        channels (list): list of #channels in each upsampling layer
        pre_residuals (int, optional): number of residual blocks before upsampling. Default: 64
        down_conv_channels (list): list of #channels in each down_conv blocks
        up_residuals (int, optional): number of residual blocks in each upsampling module. Default: 0
    """
    def __init__(self, input_len, channels, #[1025 * 2, 1024, 512, 256, 128, 64, 32, 16, 8]
          pre_residuals=4,#64,
          pre_conv_channels=[1, 1, 2],#[64, 32, 16, 8, 4],
          up_residuals=0,
          b_maxout = False,
          post_residuals=12,
          pow_2_channels=False
          ):
        super(HeadBS1, self).__init__()
        # Initialize learnable output factor
        self.f = torch.nn.Parameter(torch.ones(1))
        # Create pre_conv layer
        pre_convs = []
        c0 = pre_conv_channels[0]#1
        bs_channels = 2
        pre_convs.append(ConvBlock(bs_channels, c0, kernel_size=3, padding=1))
        for _ in range(pre_residuals):
            pre_convs.append(ResnetBlock(c0, c0))

        for i in range(len(pre_conv_channels) -1):
            in_c = pre_conv_channels[i]
            out_c = pre_conv_channels[i + 1]
            pre_convs.append(ResnetBlock(in_c, out_c))
            for _ in range(pre_residuals):
                pre_convs.append(ResnetBlock(out_c, out_c))
        self.pre_conv = nn.Sequential(*pre_convs)
        
        # Create middle layer
        mid_layers = []
        #print('start poooling')
        ch0 = pre_conv_channels[-1] * input_len # 8*100=800
        if pow_2_channels:
            get_closest_pow2_d = lambda x: np.power(2, int(np.log(x) / np.log(2)))
            ch1 = get_closest_pow2_d(ch0)#512, 256, 128, 64, 32, 16, 8
            mid_layers.append(ConvBlock(ch0, ch1, kernel_size=3, padding=1, one_d=True))
            #print(f'ConvBlock {ch0}, {ch1}')

            while ch1/2. > 0:
                in_channels = ch1
                out_channels = int(ch1/2.)
                ch1 = int(ch1/2.)
                layer = MidLayer(in_channels, out_channels, residuals=up_residuals, b_maxout=b_maxout)
                mid_layers.append(layer)
                #print(f'mid_layers {in_channels}, {out_channels}')
                if int(ch1) == 8:
                    break
        else:
            ch1 = 8
            mid_layers.append(ConvBlock(ch0, ch1, kernel_size=3, padding=1, one_d=True))
           
        self.mid = nn.Sequential(*mid_layers)
        
        # Create post layer
        post_convs = []
        last_channels = int(ch1)
        for i in range(post_residuals):
            post_convs.append(ResnetBlock(last_channels, last_channels, one_d=True, kernel_size=5))
        self.post_conv = nn.Sequential(*post_convs)

    def forward(self, x):
        """
        forward pass
        Args:
            x (Tensor): B x C x T # 100X100X2

        Returns:
            Tensor: B x C x (2^#channels * T) # 100X100X(2^#channels * 2)
        """
        #print(x.shape)
        x = x.unsqueeze(0) # reshape to [B x 2 x 100 x 100]
        #print(x.shape)

        x = self.pre_conv(x)
        #print(x.shape)

        s1, _, _, s4 = x.shape

        x = x.reshape(s1, -1, s4)
        #print(x.shape)

        x = self.mid(x)
        #print(x.shape)
        x *= self.f
        
        x2 = self.post_conv(x)
        #print(x2.shape)
        x2 *= self.f
        
        return x, x2# pre, post
        
        
        
        
        
class CNNBS1(nn.Module):
    """CNNBS1  - CNN for BS inversion - original one with reshape size in Head

    Args:
        n_heads (int): Number of heads
        layer_channels (list): list of #channels of each layer
    """
    def __init__(self, input_len=100, n_heads=3, 
         channels=[100 * 4 * 2, 512, 256, 128, 64, 32, 16, 8],
         b_maxout = False,
         pre_conv_channels=[2,2,4],
         pre_residuals=4, 
         up_residuals=0,
         post_residuals=12,
         pow_2_channels=False):
        # channels=[]
        # bs_channels = 2
        # ch1= int(input_len * pre_conv_channels[-1] * bs_channels)
        # channels.append(ch1)
        # ch2 = int(np.sqrt(np.power(2, np.floor(np.log(ch1)/np.log(2)))))
        # channels.append(ch2)
        # while ch2/2. > 0:
        #     ch2 /= 2.
        #     channels.append(int(ch2))
        #     if int(ch2) == 8:
        #        break
        # print(len(channels))
        super(CNNBS1, self).__init__()
        self.n_heads = n_heads
        self.linear = nn.Linear(channels[-1], 1)
        self.act_fn = nn.LeakyReLU()#nn.Softsign()
        self.heads = nn.ModuleList([HeadBS1(input_len, channels, b_maxout=b_maxout,
                pre_conv_channels=pre_conv_channels, 
                pre_residuals=pre_residuals, up_residuals=up_residuals,
                post_residuals=post_residuals, pow_2_channels=pow_2_channels)
                                    for _ in range(n_heads)])

    def forward(self, x):

        # Pass over Heads in "parallel"
        if self.n_heads > 1:
            pre_list = []
            post_list = []
            #print('Pass over Heads in parallel- start')

            for head in self.heads:
                pre, post = head(x)

                pre_list.append(pre)
                post_list.append(post)
                #print(head.f)
            #print('Pass over Heads in parallel- end')
            # Sum Heads outputs
            pre = torch.sum(torch.stack(pre_list), dim=0)
            post = torch.sum(torch.stack(post_list), dim=0)
        else:
            # Pass over Head
            pre, post = self.heads[0](x)
            
        # Pre output
        rs0 = self.linear(pre.transpose(1, 2))
        rs0 = self.act_fn(rs0).squeeze(-1)
        
        # Post output - actually used
        rs1 = self.linear(post.transpose(1, 2))
        rs1 = self.act_fn(rs1).squeeze(-1)
        
        return rs0, rs1 