import os
import torch
from torch.autograd import Variable
import numpy as np
import torch.nn.functional as F
import torchvision
from torchvision import transforms
import torch.optim as optim
from torch import nn
from scipy.misc import imsave


class Encoder(torch.nn.Module):
    def __init__(self, sample_size):
        super(Encoder, self).__init__()
        self.sample_size = sample_size
        self.eh_dim = 128
        self.linear1 = torch.nn.Linear(self.sample_size + 10, self.eh_dim)
        #self.relu = torch.nn.ReLU()

    def forward(self, x):
        return self.linear1(x)
        #return self.relu(self.linear1(input))

class Sampler(torch.nn.Module):
    def __init__(self, latent_size):
        super(Sampler, self).__init__()
        self.latent_size = latent_size
        self.eh_dim = 128
        self.linear_mu = torch.nn.Linear(self.eh_dim, self.latent_size)
        self.linear_sigma = torch.nn.Linear(self.eh_dim, self.latent_size)

    def forward(self, x):
        self.mu = self.linear_mu(x)
        log_sigma = self.linear_sigma(x)
        self.sigma = torch.exp(log_sigma)
        std_z = torch.from_numpy(np.random.normal(0, 1, size=self.sigma.size())).float().cuda()
        return self.mu + self.sigma * Variable(std_z, requires_grad=False).cuda()

class Decoder(torch.nn.Module):
    def __init__(self, sample_size, latent_size):
        super(Decoder, self).__init__()
        self.sample_size = sample_size
        self.latent_size = latent_size
        self.dh_dim = 128
        self.linear1 = torch.nn.Linear(self.latent_size + 10, self.dh_dim)
        self.linear2 = torch.nn.Linear(self.dh_dim, self.sample_size)
        self.relu = torch.nn.ReLU()
        self.sigmoid = torch.nn.Sigmoid()

    def forward(self, x):
        x = self.relu(self.linear1(x))
        return self.sigmoid(self.linear2(x))


def genYVec(y):
    y_vec = list()
    for i in y:
        mask = [0. for j in range(10)]
        mask[i] = 1.
        y_vec.append(mask)
    return np.asarray(y_vec)


if __name__ == '__main__':
    sample_size = 784 #28 * 28
    mb_size = 32
    z_dim = 100

    transform = transforms.Compose([transforms.ToTensor()])
    mnist = torchvision.datasets.MNIST('./', download=True, transform=transform)
    dataloader = torch.utils.data.DataLoader(mnist, batch_size=mb_size, shuffle=True, num_workers=2)

    imgs_folder = './out'
    if not os.path.exists(imgs_folder):
        os.makedirs(imgs_folder)
    
    encoder = Encoder(sample_size)
    encoder = encoder.cuda()
    sampler = Sampler(z_dim)
    sampler = sampler.cuda()
    decoder = Decoder(sample_size, z_dim)
    decoder = decoder.cuda()
    params = []
    params.extend(list(encoder.parameters()))
    params.extend(list(sampler.parameters()))
    params.extend(list(decoder.parameters()))
    optimizer = optim.Adam(params, lr=0.0001)
    #optimizer = optim.Adam(encoder.parameters(), lr=0.0001)
    #criterion = nn.MSELoss()
    for epoch in range(100):
        for i, data in enumerate(dataloader, 0):
            x_mb, y_mb = data
            y = genYVec(y_mb)
            y = torch.Tensor(y)
            raw = x_mb.resize_(mb_size, sample_size)
            x_mb = Variable(torch.cat((x_mb, y), 1), requires_grad=False).cuda()
            #x_mb = torch.cat((x_mb, y), 1)
            enc = encoder(x_mb)
            z = sampler(enc)
            z = Variable(torch.cat((z.cpu().data, y), 1), requires_grad=False).cuda()
            dec = decoder(z)
            kl_loss = 0.5 * torch.mean(sampler.mu*sampler.mu + sampler.sigma*sampler.sigma - torch.log(sampler.sigma*sampler.sigma) -1)
            #recon_loss = criterion(dec, x_mb)
            recon_loss = F.binary_cross_entropy(dec, Variable(raw, requires_grad=False).cuda())
            loss = recon_loss + kl_loss
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            l = loss.data[0]
            if i % 1000 == 0:
                print 'Epoch:%d, iteration:%d, loss:%f' % (epoch, i, l)
                z = torch.from_numpy(np.random.normal(0, 1, size=[1, z_dim])).float().cuda()
                mask = [0. for j in range(10)]
                mask[0] = 1.
                mask = torch.from_numpy(np.asarray([mask])).float().cuda()
                z = Variable(torch.cat((z, mask), 1), requires_grad=False).cuda()
                imgs = decoder(z).cpu().data.numpy()
                imsave(os.path.join(imgs_folder, '%d_%d.png') % (epoch, i), imgs[0].reshape(28, 28))
            
