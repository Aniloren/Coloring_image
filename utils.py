from colorization_dataset import ColorizationDataset
from torch.utils.data import DataLoader
from torchvision.models.resnet import resnet18
from torch import nn
import torch
from skimage.color import rgb2lab, lab2rgb
import matplotlib.pyplot as plt
import time
import numpy as np
from fastai.vision.models.unet import DynamicUnet
from fastai.vision.learner import create_body
from PIL import Image
from torchvision import transforms
from mainmodel import MainModel


class AverageMeter:
    def __init__(self):
        self.reset()

    def reset(self):
        self.count, self.avg, self.sum = [0.] * 3

    def update(self, val, count=1):
        self.count += count
        self.sum += count * val
        self.avg = self.sum / self.count


def make_dataloaders(batch_size=32, n_workers=3, pin_memory=True, **kwargs):  # A handy function to make our dataloaders
    dataset = ColorizationDataset(**kwargs)
    dataloader = DataLoader(dataset, batch_size=batch_size, num_workers=n_workers,
                            pin_memory=pin_memory)
    return dataloader


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
    fig, axs = plt.subplots(3, n_col)
    if n_col == 1:
        for j in range(n_col):
            axs[0].imshow(L[j][0].cpu(), cmap='gray')
            axs[0].axis("off")
            axs[1].imshow(fake_imgs[j])
            axs[1].axis("off")
            axs[2].imshow(real_imgs[j])
            axs[2].axis("off")
    else:
        for j in range(n_col):
            # for i in range(3):
            print('j ', j, n_col)
            axs[0, j].imshow(L[j][0].cpu(), cmap='gray')
            axs[0, j].axis("off")
            axs[1, j].imshow(fake_imgs[j])
            axs[1, j].axis("off")
            axs[2, j].imshow(real_imgs[j])
            axs[2, j].axis("off")

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


def build_res_unet(n_input=1, n_output=2, size=256):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    body = create_body(resnet18, pretrained=True, n_in=n_input, cut=-2)
    net_G = DynamicUnet(body, n_output, (size, size)).to(device)
    return net_G


def make_colore(filename):
    fake = Image.open(filename)
    transform = transforms.Resize((256, 256), Image.BICUBIC)
    fake = transform(fake)
    fake = np.array(fake)
    fake_lab = rgb2lab(fake).astype("float32")
    fake_lab = transforms.ToTensor()(fake_lab)

    fake_L = fake_lab[[0], ...] / 50. - 1.
    fake_L = torch.reshape(fake_L, (1, 1, 256, 256)).cuda()
    net_G = build_res_unet(n_input=1, n_output=2, size=256)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net_G.load_state_dict(torch.load("weights/res18-unet.pt", map_location=device))
    model = MainModel(net_G=net_G)
    model.load_state_dict(torch.load("weights/final_model_weights.pt", map_location=device))

    print('load done')

    res = model.forward_fake(fake_L)
    res = res.cpu().detach()
    result = lab_to_rgb(fake_L.cpu(), res)

    filename2 = 'filename.jpg'
    plt.axis('off')
    plt.imshow(result[0])
    plt.savefig(filename2, bbox_inches='tight', pad_inches=0)
    print("SAVE IMAGE DONE")

    return filename2