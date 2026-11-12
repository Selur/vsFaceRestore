"""Optical flow for KEEP: the GMFlow wrapper (KEEP `basicsr/archs/gmflow_arch.py`) and the flow helpers
`flow_warp` / `resize_flow` (KEEP `basicsr/archs/arch_util.py`, from BasicSR). The GMFlow network itself is the
`gmflow` package next to this file (haofeixu/gmflow, commit b512343, Apache License 2.0).
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .gmflow.gmflow import GMFlow


class FlowGenerator(nn.Module):
    """GM flow generation.

    Args:
        path (str): Pre-trained path. Default: None.
        requires_grad (bool): If true, the parameters of VGG network will be
            optimized. Default: False.
    """

    def __init__(self,
                 path=None,
                 requires_grad=False,):
        super().__init__()

        self.model = GMFlow()

        if path != None:
            weights = torch.load(
                path, map_location=lambda storage, loc: storage)['model']
            self.model.load_state_dict(weights, strict=True)

        if not requires_grad:
            self.model.eval()
            for param in self.parameters():
                param.requires_grad = False
        else:
            self.model.train()
            for param in self.parameters():
                param.requires_grad = True

    def forward(self, im1, im2,
                attn_splits_list=[2],
                corr_radius_list=[-1],
                prop_radius_list=[-1]):
        """Forward function.

        Args:
            im1 (Tensor): Input tensor with shape (n, c, h, w).
            im2 (Tensor): Input tensor with shape (n, c, h, w).

        Returns:
            Tensor: Forward results.
        """
        assert im1.shape == im2.shape
        N, C, H, W = im1.shape

        im1 = (im1 + 1) / 2 * 255
        im2 = (im2 + 1) / 2 * 255

        flow = self.model(im1, im2,
                          attn_splits_list=attn_splits_list,
                          corr_radius_list=corr_radius_list,
                          prop_radius_list=prop_radius_list,
                          pred_bidir_flow=False)['flow_preds'][-1]
        # backward_flow = flow[N:]

        return flow


def flow_warp(x, flow, interp_mode='bilinear', padding_mode='zeros', align_corners=True):
    """Warp an image or feature map with optical flow.

    Args:
        x (Tensor): Tensor with size (n, c, h, w).
        flow (Tensor): Tensor with size (n, h, w, 2), normal value.
        interp_mode (str): 'nearest' or 'bilinear'. Default: 'bilinear'.
        padding_mode (str): 'zeros' or 'border' or 'reflection'.
            Default: 'zeros'.
        align_corners (bool): Before pytorch 1.3, the default value is
            align_corners=True. After pytorch 1.3, the default value is
            align_corners=False. Here, we use the True as default.

    Returns:
        Tensor: Warped image or feature map.
    """
    assert x.size()[-2:] == flow.size()[1:3]
    _, _, h, w = x.size()
    # create mesh grid
    grid_y, grid_x = torch.meshgrid(torch.arange(0, h).type_as(x), torch.arange(0, w).type_as(x))
    grid = torch.stack((grid_x, grid_y), 2).float()  # W(x), H(y), 2
    grid.requires_grad = False

    vgrid = grid + flow
    # scale grid to [-1,1]
    vgrid_x = 2.0 * vgrid[:, :, :, 0] / max(w - 1, 1) - 1.0
    vgrid_y = 2.0 * vgrid[:, :, :, 1] / max(h - 1, 1) - 1.0
    vgrid_scaled = torch.stack((vgrid_x, vgrid_y), dim=3)
    output = F.grid_sample(x, vgrid_scaled, mode=interp_mode, padding_mode=padding_mode, align_corners=align_corners)

    # TODO, what if align_corners=False
    return output


def resize_flow(flow, size_type, sizes, interp_mode='bilinear', align_corners=False):
    """Resize a flow according to ratio or shape.

    Args:
        flow (Tensor): Precomputed flow. shape [N, 2, H, W].
        size_type (str): 'ratio' or 'shape'.
        sizes (list[int | float]): the ratio for resizing or the final output
            shape.
            1) The order of ratio should be [ratio_h, ratio_w]. For
            downsampling, the ratio should be smaller than 1.0 (i.e., ratio
            < 1.0). For upsampling, the ratio should be larger than 1.0 (i.e.,
            ratio > 1.0).
            2) The order of output_size should be [out_h, out_w].
        interp_mode (str): The mode of interpolation for resizing.
            Default: 'bilinear'.
        align_corners (bool): Whether align corners. Default: False.

    Returns:
        Tensor: Resized flow.
    """
    _, _, flow_h, flow_w = flow.size()
    if size_type == 'ratio':
        output_h, output_w = int(flow_h * sizes[0]), int(flow_w * sizes[1])
    elif size_type == 'shape':
        output_h, output_w = sizes[0], sizes[1]
    else:
        raise ValueError(f'Size type should be ratio or shape, but got type {size_type}.')

    input_flow = flow.clone()
    ratio_h = output_h / flow_h
    ratio_w = output_w / flow_w
    input_flow[:, 0, :, :] *= ratio_w
    input_flow[:, 1, :, :] *= ratio_h
    resized_flow = F.interpolate(
        input=input_flow, size=(output_h, output_w), mode=interp_mode, align_corners=align_corners)
    return resized_flow
