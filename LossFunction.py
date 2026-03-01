import torch.nn as nn
import torch
import torch.nn.functional as F
import lpips
from torch.nn.functional import normalize
import matplotlib.pyplot as plt
from torchvision.utils import save_image

class Loss_Function(nn.Module):
    def __init__(self, l1_weight=10.0, ssim_weight=1.0, tv_weight=10.0, lpips_weight=2.0, hist_weight=1.0, normal_weight=30.0):
        super(Loss_Function, self).__init__()
        self.cosine_loss = nn.CosineSimilarity()
        self.l1_loss = nn.L1Loss()
        self.SSIM = SSIM()
        self.LPIPS = lpips.LPIPS(net='vgg')
        self.l1_weight = l1_weight
        self.ssim_weight = ssim_weight
        self.tv_weight = tv_weight
        self.lpips_weight = lpips_weight
        self.hist_weight = hist_weight
        self.normal_weight = normal_weight

    def forward(self, predict, ground_truth, descattering_img, CleanWater, normal_hist, mask, train_loader):
        # descattering loss
        descattering_img = descattering_img * mask
        CleanWater = CleanWater * mask
        L_1 = self.l1_loss(descattering_img, CleanWater)
        L_ssim = 1 - self.SSIM(descattering_img, CleanWater)
        L_tv = total_variation_loss(descattering_img)
        # The input of LPIPS must be a 3-channel image, where the Imean average intensity image is equivalently expanded to 3-channel
        Imean_pred = four_pol_to_mean_intensity(descattering_img)
        Imean_gt   = four_pol_to_mean_intensity(CleanWater)
        # LPIPS requires an input range of [-1,1]
        Imean_pred = Imean_pred * 2 - 1
        Imean_gt   = Imean_gt * 2 - 1
        L_lpips = self.LPIPS(Imean_pred, Imean_gt).mean()

        # histogram loss
        GT_hist = convert_images_to_color_hist_tensor(ground_truth, hist_size=64).cuda()
        L_hist = self.l1_loss(normal_hist, GT_hist)

        # normal loss
        predict = predict * mask
        # predict = normalize(predict, dim=1)
        ground_truth = ground_truth * mask
        ground_truth_n = (ground_truth * 2.0 -1.0) * mask
        cosine = 1 - self.cosine_loss(predict, ground_truth_n)
        num_cosine = torch.sum(torch.sum(torch.sum(cosine, dim=1), dim=1))
        M = torch.sum(torch.sum(torch.sum(mask, dim=1), dim=1))  # Foreground object pixels
        back_ground = (train_loader.batch_size * 256 * 256) - M  # Background region pixels
        loss_cosine = num_cosine - back_ground
        L_normal = loss_cosine / M

        # total loss
        total_loss = self.l1_weight * L_1 + self.ssim_weight * L_ssim + self.tv_weight * L_tv + self.lpips_weight * L_lpips + self.hist_weight * L_hist + self.normal_weight * L_normal
        # print("L_descattering loss: ", self.descattering_weight * L_descattering.item())
        # print("L_ssim loss:", self.ssim_weight * L_ssim.item())
        # print("L_tv loss:", self.tv_weight * L_tv.item())
        # print("L_lpips loss:", self.lpips_weight * L_lpips.item())
        # print("L_hist loss: ", self.hist_weight * L_hist.item())
        # print("L_normal loss: ", self.normal_weight * L_normal.item())
        # visualization_tensor(descattering_img, normal_hist, 'descattering_img', 'normal_hist')
        # visualization_tensor(CleanWater, GT_hist, 'CleanWater', 'GT_hist')
        return total_loss

def four_pol_to_mean_intensity(p4, eps=1e-6):
    # p4: [N,4,H,W] -> I0, I45, I90, I135
    I0  = p4[:, 0, :, :]
    I45 = p4[:, 1, :, :]
    I90 = p4[:, 2, :, :]
    I135= p4[:, 3, :, :]

    I_mean = (I0 + I45 + I90 + I135) / 4

    # Normalize to [0,1] for easy scale alignment with LPIPS
    I_mean = (I_mean - I_mean.min()) / (I_mean.max() - I_mean.min() + eps)
    I_mean = torch.stack([I_mean, I_mean, I_mean], dim=1)

    return I_mean


def convert_images_to_color_hist_tensor(image_tensor_batch, hist_size=64):
    """
    Args:
        image_tensor_batch: Tensor (B, 3, H, W), pixel in [0, 1] or [-1,1], 已被掩码处理
    Returns:
        Tensor: (B, 3, hist_size)
    """
    # image_tensor_batch = (image_tensor_batch * 2) - 1.0

    B, C, H, W = image_tensor_batch.shape
    hist_list = []

    for img in image_tensor_batch:  # img shape: (3, H, W)
        channel_hists = []

        for ch in range(C):
            channel = img[ch]  # (H, W)
            # Filter out black pixels (value 0.0)
            valid_pixels = channel[channel > 0]

            normal_coordinates = (channel * 2) - 1

            hist = torch.histc(normal_coordinates, bins=hist_size, min=-1.0, max=1.0)
            hist = hist / (H * W)
            channel_hists.append(hist)

        hist_list.append(torch.stack(channel_hists))  # (3, hist_size)

    return torch.stack(hist_list).to(image_tensor_batch.device)  # (B, 3, hist_size)

def _ssim(img1, img2, window, window_size, channel, size_average=True):
    mu1 = F.conv2d(img1, window, padding=window_size // 2, groups=channel)
    mu2 = F.conv2d(img2, window, padding=window_size // 2, groups=channel)
    # print(mu1.shape,mu2.shape)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = F.conv2d(img1 * img1, window, padding=window_size // 2, groups=channel) - mu1_sq
    sigma2_sq = F.conv2d(img2 * img2, window, padding=window_size // 2, groups=channel) - mu2_sq
    sigma12 = F.conv2d(img1 * img2, window, padding=window_size // 2, groups=channel) - mu1_mu2

    C1 = 0.01 ** 2
    C2 = 0.03 ** 2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))
    mcs_map = (2.0 * sigma12 + C2) / (sigma1_sq + sigma2_sq + C2)
    # print(ssim_map.shape)
    if size_average:
        return ssim_map.mean(), mcs_map.mean()
    # else:
    #     return ssim_map.mean(1).mean(1).mean(1)


class SSIM(torch.nn.Module):
    def __init__(self, window_size=11, size_average=True):
        super(SSIM, self).__init__()
        self.window_size = window_size
        self.size_average = size_average

    def forward(self, img1, img2):
        _, channel, _, _ = img1.size()
        window = create_window(self.window_size, channel)
        window = type_trans(window, img1)
        ssim_map, mcs_map = _ssim(img1, img2, window, self.window_size, channel, self.size_average)
        return ssim_map

def create_window(window_size, channel):
    # Create Gaussian window
    sigma = 1.5  # Experience-based parameter
    coords = torch.arange(window_size).float() - window_size // 2
    g = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
    g /= g.sum()
    window = g.ger(g).unsqueeze(0).unsqueeze(0)
    return window.repeat(channel, 1, 1, 1)

def type_trans(window, img):
    if img.is_cuda:
        window = window.cuda(img.get_device())
    return window.type_as(img)

def total_variation_loss(img):
    # batch_size, channels, height, width = img.size()
    h_variation = torch.mean(torch.abs(img[:, :, :, :-1] - img[:, :, :, 1:]))
    v_variation = torch.mean(torch.abs(img[:, :, 1:, :] - img[:, :, :-1, :]))
    loss_tv = (h_variation + v_variation)
    return loss_tv

def visualization_tensor(descattering_img, normal_hist, descattering_name, normal_hist_name):
    descattering_img = descattering_img.cpu().detach()  # shape: (2, 4, 256, 256)
    normal_hist = normal_hist.cpu().detach().numpy()  # (2, 3, 64)

    for b in range(descattering_img.shape[0]):
        for c in range(descattering_img.shape[1]):
            img = descattering_img[b, c:c + 1]  # (1, H, W)，Grayscale images should retain single channel format
            save_image(img, f'./results_sfp/{descattering_name}_{c}.bmp')
    for b in range(normal_hist.shape[0]):
        plt.figure(figsize=(12, 4))
        for c in range(3):  # R, G, B
            plt.subplot(1, 3, c + 1)
            plt.bar(range(64), normal_hist[b, c])
            plt.title(f'Batch {b} - Channel {["R", "G", "B"][c]}')
            plt.xlabel('Bin')
            plt.ylabel('Frequency')
        plt.tight_layout()
        plt.savefig(f'./results_sfp/{normal_hist_name}.png')
        plt.close()
