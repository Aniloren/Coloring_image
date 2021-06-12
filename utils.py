from colorization_dataset import ColorizationDataset
from torch.utils.data import DataLoader
from torch import nn
import torch
from skimage.color import rgb2lab, lab2rgb
import matplotlib.pyplot as plt
import time
import numpy as np

def make_dataloaders(batch_size=32, n_workers=3, pin_memory=True, **kwargs): # A handy function to make our dataloaders
    dataset = ColorizationDataset(**kwargs)
    dataloader = DataLoader(dataset, batch_size=batch_size, num_workers=n_workers,
                            pin_memory=pin_memory)
    return dataloader

def _init_weights(net, init='norm', gain=0.02):
    
    def init_func(m):
        classname = m.__class__.__name__
        if hasattr(m, 'weight') and 'Conv' in classname:
            if init == 'norm':
                nn.init.normal_(m.weight.data, mean=0.0, std=gain)
            elif init == 'xavier':
                nn.init.xavier_normal_(m.weight.data, gain=gain)
            elif init == 'kaiming':
                nn.init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
            
            if hasattr(m, 'bias') and m.bias is not None:
                nn.init.constant_(m.bias.data, 0.0)
        elif 'BatchNorm2d' in classname:
            nn.init.normal_(m.weight.data, 1., gain)
            nn.init.constant_(m.bias.data, 0.)
            
    net.apply(init_func)
    print(f"model initialized with {init} initialization")
    return net

def init_model(model, device):
    model = model.to(device)
    model = _init_weights(model)
    return model


class AverageMeter:
    def __init__(self):
        self.reset()
        
    def reset(self):
        self.count, self.avg, self.sum = [0.] * 3
    
    def update(self, val, count=1):
        self.count += count
        self.sum += count * val
        self.avg = self.sum / self.count

def create_loss_meters():
    loss_D_fake = AverageMeter()
    loss_D_real = AverageMeter()
    loss_D = AverageMeter()
    loss_G_GAN = AverageMeter()
    loss_G_L1 = AverageMeter()
    loss_G = AverageMeter()
    
    return {'loss_D_fake': loss_D_fake,
            'loss_D_real': loss_D_real,
            'loss_D': loss_D,
            'loss_G_GAN': loss_G_GAN,
            'loss_G_L1': loss_G_L1,
            'loss_G': loss_G}

def update_losses(model, loss_meter_dict, count):
    for loss_name, loss_meter in loss_meter_dict.items():
        loss = getattr(model, loss_name)
        loss_meter.update(loss.item(), count=count)

def lab_to_rgb(L, ab):
    """
    Takes a batch of images
    """
    
    L = (L + 1.) * 50.
    ab = ab * 110.
    Lab = torch.cat([L, ab], dim=1).permute(0, 2, 3, 1).cpu().numpy()
    rgb_imgs = []
    for img in Lab:
        img_rgb = lab2rgb(img)
        rgb_imgs.append(img_rgb)
    return np.stack(rgb_imgs, axis=0)
    
def visualize(model, data, save=True):
    model.net_G.eval()
    with torch.no_grad():
        model.setup_input(data)
        model.forward()
    model.net_G.train()
    fake_color = model.fake_color.detach()
    real_color = model.ab
    L = model.L
    fake_imgs = lab_to_rgb(L, fake_color)
    real_imgs = lab_to_rgb(L, real_color)
    fig = plt.figure(figsize=(15, 8))
    print('data:', len(data['L']))
    n_col = len(data['L'])
    if n_col > 6:
        n_col = 6
    fig, axs = plt.subplots(3,  n_col)
    if n_col == 1:
        for j in range(n_col):
        # for i in range(3):
        # print('j ',j, n_col)
            axs[0].imshow(L[j][0].cpu() , cmap='gray')
            axs[0].axis("off")
            axs[1].imshow(fake_imgs[j])
            axs[1].axis("off")
            axs[2].imshow(real_imgs[j])
            axs[2].axis("off")
    else:
        for j in range(n_col):
            # for i in range(3):
            print('j ',j, n_col)
            axs[0,j].imshow(L[j][0].cpu() , cmap='gray')
            axs[0,j].axis("off")
            axs[1,j].imshow(fake_imgs[j])
            axs[1,j].axis("off")
            axs[2,j].imshow(real_imgs[j])
            axs[2,j].axis("off")

    
    plt.show()
    # for i in range(5):
    #     ax = plt.subplot(3, 5, i + 1)
    #     ax.imshow(L[i][0].cpu(), cmap='gray')
    #     ax.axis("off")
    #     ax = plt.subplot(3, 5, i + 1 + 5)
    #     ax.imshow(fake_imgs[i])
    #     ax.axis("off")
    #     ax = plt.subplot(3, 5, i + 1 + 10)
    #     ax.imshow(real_imgs[i])
    #     ax.axis("off")
    # plt.show()
    if save:
        fig.savefig(f"colorization_{time.time()}.png")
        
def log_results(loss_meter_dict):
    for loss_name, loss_meter in loss_meter_dict.items():
        print(f"{loss_name}: {loss_meter.avg:.5f}")