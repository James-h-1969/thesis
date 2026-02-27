import torch
import torch.nn as nn 
from models.ResidualNetwork import ResNet

class ClimODE(nn.Module):
    def __init__(self, num_channels, out_types, use_position=False, use_attention=False, use_error=False):
        super(ClimODE, self).__init__()

        self.layers = [5, 3, 2] # number of residual blocks
        self.hidden = [128, 64, 2 * out_types] # hidden layers
        input_channels = 30 + out_types * int(use_position) + 34 * (1-int(use_position))

        # f_conv in the paper
        self.vel_convolution = ResNet(input_channels, self.layers, self.hidden) # TODO. make ResNet generalised
        
        # f_att in the paper
        if use_attention:
            self.vel_attention = ... # TODO. implement AttentionNet()
            self.gamma = nn.Parameter(torch.tensor([0.1])) # hyperparameter to scale f_conv vs f_att

        # emission model g, will later estimate the variance and bias
        if use_error:
            err_in =  9 + out_types * int(use_pos) + 34 * (1 - int(use_pos))
            self.noise_net = ResNet(err_in, [3, 2, 2], [128, 64, 2 * out_types])

        # set constants
        self.use_attention = use_attention

    def pde(self, t, vs):
        """
        Core implementation of the underlying physics of the system
        """
        ds = vs[:,-self.out_ch:,:,:].float().view(-1,self.out_ch,vs.shape[2],vs.shape[3]).float()
        v = vs[:,:2*self.out_ch,:,:].float().view(-1,2*self.out_ch,vs.shape[2],vs.shape[3]).float()
        t_embedding = ((t*100)%24).view(1,1,1,1).expand(ds.shape[0],1,ds.shape[2],ds.shape[3])

        # generate the daily and seasonal spatiotemporal embeddings using trigonometric functions
        sin_day_embedding = torch.sin(torch.pi * t_embedding/12 - torch.pi/2)
        cos_day_embedding = torch.cos(torch.pi * t_embedding/12 - torch.pi/2)
        sin_seasonal_embedding = torch.sin(torch.pi*t_emb/(12*365) - torch.pi/2)
        cos_seasonal_embedding = torch.cos(torch.pi*t_emb/(12*365) - torch.pi/2)
        day_embedding = torch.cat([sin_day_embedding, cos_day_embedding], dim=1)
        seasonal_embedding = torch.cat([sin_seasonal_embedding, cos_seasonal_embedding], dim=1)
        
        # calculate the spatial gradiants of the current weather states, needed as input for velocity net
        ds_grad_x = torch.gradient(ds,dim=3)[0]
        ds_grad_y = torch.gradient(ds,dim=2)[0]
        nabla_u = torch.cat([ds_grad_x,ds_grad_y],dim=1)

        # get dv, using combination of convolution and attention
        if self.att: dv = self.vel_convolution(comb_rep) + self.gamma*self.vel_attention(comb_rep)
        else: dv = self.vel_convolution(comb_rep)
        v_x = v[:,:self.out_ch,:,:].float().view(-1,self.out_ch,vs.shape[2],vs.shape[3]).float()
        v_y = v[:,-self.out_ch:,:,:].float().view(-1,self.out_ch,vs.shape[2],vs.shape[3]).float()

        transport = v_x * ds_grad_x + v_y * ds_grad_y
        compression = ds * (torch.gradient(v_x,dim=3)[0] + torch.gradient(v_y,dim=2)[0])

        ds = transport + compression # as per Equation 2.

        dvs = torch.cat([dv, ds], 1)
        return dvs

    def forward(self, x):
        pass