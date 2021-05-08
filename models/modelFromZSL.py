##ref https://github.com/EthanZhu90/ZSL_GAN/blob/master/models.py

import torch
import torch.nn as nn

import cfg.configs as configs

class _param:
    def __init__(self):
        self.rdc_text_dim = configs.rdc_text_dim
        self.z_dim = configs.z_dim
        self.h_dim = configs.h_dim

# reduce to dim of text first
class _netG(nn.Module):
    def __init__(self, text_dim=11083, X_dim=3584):
        super(_netG, self).__init__()
        self.rdc_text = nn.Linear(text_dim, configs.rdc_text_dim)
        self.main = nn.Sequential(nn.Linear(configs.z_dim + configs.rdc_text_dim, configs.h_dim),
                                  nn.LeakyReLU(),
                                  nn.Linear(configs.h_dim, X_dim),
                                  nn.Tanh())

    def forward(self, z, c):
        rdc_text = self.rdc_text(c)
        input = torch.cat([z, rdc_text], 1)
        output = self.main(input)
        return output

class DEncoder(nn.Module):
    def __init__(self, indim, outdim):
        super(DEncoder, self).__init__()
        self.d1 = nn.Linear(indim, outdim)
        self.d2 = nn.Linear(outdim, outdim)

    def forward(self, x):
        x = self.d1(x)
        x = nn.LeakyReLU()(x)
        x = self.d2(x)
        x = nn.LeakyReLU()(x)
        return x


class _netD(nn.Module):
    def __init__(self, y_dim=150, X_dim=3584):
        super(_netD, self).__init__()
        # Discriminator net layer one
        self.D_shared = nn.Sequential(nn.Linear(X_dim, configs.h_dim),
                                      nn.ReLU())
        # Discriminator net branch one: For Gan_loss
        self.D_gan = nn.Linear(configs.h_dim, 1)
        # Discriminator net branch two: For aux cls loss
        self.D_aux = nn.Linear(configs.h_dim, y_dim)

    def forward(self, input):
        h = self.D_shared(input)
        return self.D_gan(h), self.D_aux(h)

# In GBU setting, using attribute
class _netG_att(nn.Module):
    def __init__(self, opt, att_dim, X_dim):
        super(_netG_att, self).__init__()
        self.main = nn.Sequential(nn.Linear(opt.z_dim + att_dim, configs.h_dim),
                                  nn.LeakyReLU(),
                                  nn.Linear(configs.h_dim, X_dim),
                                  nn.Tanh())
    def forward(self, z, c):
        input = torch.cat([z, c], 1)
        output = self.main(input)
        return output
