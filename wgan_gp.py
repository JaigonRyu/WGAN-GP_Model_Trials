
from torch import nn
import torch.nn.utils as nn_utils

class Generator(nn.Module):
    def __init__(self, ngpu, z_dim,hidden_dim, im_chan=3):
        super(Generator, self).__init__()
        self.ngpu = ngpu
        self.z_dim = z_dim

        self.main = nn.Sequential(
           
            nn.ConvTranspose2d(z_dim, hidden_dim * 8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(hidden_dim * 8),
            nn.ReLU(True),


            nn.ConvTranspose2d(hidden_dim * 8, hidden_dim * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(hidden_dim * 4),
            nn.ReLU(True),

    
            nn.ConvTranspose2d(hidden_dim * 4, hidden_dim * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(hidden_dim * 2),
            nn.ReLU(True),

            nn.ConvTranspose2d(hidden_dim * 2, im_chan, 4, 2, 1, bias=False),
            nn.Tanh()
        )

    def forward(self, input):
        x = input.view(len(input), self.z_dim, 1, 1)
        return self.main(x)



    
class Critic(nn.Module):
    def __init__(self, hidden_dim, im_chan=3):
        super(Critic, self).__init__()

        self.critic = nn.Sequential(
         
            nn.Conv2d(im_chan, hidden_dim, kernel_size=4, stride=2),
            nn.LeakyReLU(0.2, inplace=True),

       
            nn.Conv2d(hidden_dim, hidden_dim * 2, kernel_size=4, stride=2),
            nn.LeakyReLU(0.2, inplace=True),

            
            nn.Conv2d(hidden_dim * 2, 1, kernel_size=4, stride=2)
        )

    def forward(self, image):
        crit_pred = self.critic(image)
        return crit_pred.view(len(crit_pred), -1)  # flatten the output


class CriticSpec(nn.Module):
    def __init__(self, hidden_dim, im_chan=3):
        super(CriticSpec, self).__init__()

        self.critic = nn.Sequential(
            nn_utils.spectral_norm(nn.Conv2d(im_chan, hidden_dim, kernel_size=4, stride=2)),
            nn.LeakyReLU(0.2, inplace=True),

            nn_utils.spectral_norm(nn.Conv2d(hidden_dim, hidden_dim * 2, kernel_size=4, stride=2)),
            nn.LeakyReLU(0.2, inplace=True),

            nn_utils.spectral_norm(nn.Conv2d(hidden_dim * 2, 1, kernel_size=4, stride=2))
        )

    def forward(self, image):
        crit_pred = self.critic(image)
        return crit_pred.view(len(crit_pred), -1)
    
class CriticDrop(nn.Module):
    def __init__(self, hidden_dim, im_chan=3):
        super(CriticDrop, self).__init__()

        self.critic = nn.Sequential(
            nn_utils.spectral_norm(nn.Conv2d(im_chan, hidden_dim, kernel_size=4, stride=2)),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout2d(0.3),

            nn_utils.spectral_norm(nn.Conv2d(hidden_dim, hidden_dim * 2, kernel_size=4, stride=2)),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout2d(0.3),

            nn_utils.spectral_norm(nn.Conv2d(hidden_dim * 2, 1, kernel_size=4, stride=2))
        )

    def forward(self, image):
        crit_pred = self.critic(image)
        return crit_pred.view(len(crit_pred), -1)
