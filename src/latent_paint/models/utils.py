'''
ming
'''
import torch 
import torch.nn as nn


def init_paint(linear_rgb_estimator, texture_resolution, init_rgb_color=(1.0, 0.0, 0.0)):

    # inverse linear approx to find latent
    A = linear_rgb_estimator.T
    regularizer = 1e-2
    init_color_in_latent = (torch.pinverse(A.T @ A + regularizer * torch.eye(4).cuda()) @ A.T) @ torch.tensor(
        list(init_rgb_color)).float().to(A.device)

    # init colors with target latent plus some noise
    texture_img = nn.Parameter(
        init_color_in_latent[None, :, None, None] * 0.3 + 0.4 * torch.randn(1, 4, texture_resolution,
                                                                            texture_resolution).cuda())

    # used only for latent-paint fine-tuning, values set when reading previous checkpoint statedict
    texture_img_rgb_finetune = nn.Parameter(torch.zeros(1, 3,
                                                        texture_resolution, texture_resolution).cuda())

    return texture_img, texture_img_rgb_finetune