# dot_functions.py

import torch
import numpy as np
from einops import rearrange
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))
from dot.models import create_model

##########################################################################
# Global variables to cache the DOT model and keep track of its dimensions
##########################################################################
_DOT_MODEL = None
_DOT_MODEL_WIDTH = None
_DOT_MODEL_HEIGHT = None

##########################################################################
# Color wheel and flow visualization utilities
##########################################################################

def make_colorwheel(device):
    """
    Generates a color wheel for optical flow visualization as presented in:
    Baker et al. "A Database and Evaluation Methodology for Optical Flow" (ICCV, 2007)
    Returns:
        torch.Tensor: Color wheel tensor of shape [ncols, 3], values in [0, 255]
    """
    RY = 15
    YG = 6
    GC = 4
    CB = 11
    BM = 13
    MR = 6

    ncols = RY + YG + GC + CB + BM + MR
    colorwheel = torch.zeros((ncols, 3), device=device)

    col = 0

    # RY
    colorwheel[0:RY, 0] = 255
    colorwheel[0:RY, 1] = torch.floor(255 * torch.arange(0, RY, device=device) / RY)
    col += RY

    # YG
    colorwheel[col:col+YG, 0] = 255 - torch.floor(255 * torch.arange(0, YG, device=device) / YG)
    colorwheel[col:col+YG, 1] = 255
    col += YG

    # GC
    colorwheel[col:col+GC, 1] = 255
    colorwheel[col:col+GC, 2] = torch.floor(255 * torch.arange(0, GC, device=device) / GC)
    col += GC

    # CB
    colorwheel[col:col+CB, 1] = 255 - torch.floor(255 * torch.arange(0, CB, device=device) / CB)
    colorwheel[col:col+CB, 2] = 255
    col += CB

    # BM
    colorwheel[col:col+BM, 2] = 255
    colorwheel[col:col+BM, 0] = torch.floor(255 * torch.arange(0, BM, device=device) / BM)
    col += BM

    # MR
    colorwheel[col:col+MR, 2] = 255 - torch.floor(255 * torch.arange(0, MR, device=device) / MR)
    colorwheel[col:col+MR, 0] = 255

    return colorwheel

def flow_uv_to_colors(u, v, convert_to_bgr=False):
    """
    Applies the flow color wheel to flow components u and v.
    Args:
        u (torch.Tensor): Horizontal flow of shape [N_frames, H, W]
        v (torch.Tensor): Vertical flow of shape [N_frames, H, W]
        convert_to_bgr (bool, optional): Convert output image to BGR. Defaults to False.
    Returns:
        torch.Tensor: Flow visualization images of shape [N_frames, H, W, 3], values in [0,1]
    """
    device = u.device
    n_frames, H, W = u.shape
    flow_image = torch.zeros(n_frames, H, W, 3, device=device)
    colorwheel = make_colorwheel(device)  # [ncols, 3]
    ncols = colorwheel.shape[0]

    rad = torch.sqrt(u**2 + v**2)
    a = torch.atan2(-v, -u) / np.pi
    fk = (a + 1) / 2 * (ncols - 1)
    k0 = torch.floor(fk).long()
    k1 = k0 + 1
    k1 = k1 % ncols  # wrap around
    f = fk - k0.float()

    for i in range(3):
        tmp = colorwheel[:, i]
        col0 = tmp[k0] / 255.0
        col1 = tmp[k1] / 255.0
        col = (1 - f) * col0 + f * col1

        idx = rad <= 1
        col = torch.where(idx, 1 - rad * (1 - col), col * 0.75)

        if convert_to_bgr:
            ch_idx = 2 - i
        else:
            ch_idx = i
        flow_image[..., ch_idx] = col

    return flow_image.clamp(0, 1)

def flow_to_color(flow_uv, clip_flow=None, convert_to_bgr=False):
    """
    Converts flow UV images to RGB images.
    Args:
        flow_uv (torch.Tensor): Flow UV images of shape [N_frames, H, W, 2]
        clip_flow (float, optional): Clip maximum of flow values. Defaults to None.
        convert_to_bgr (bool, optional): Convert output image to BGR. Defaults to False.
    Returns:
        torch.Tensor: RGB images of shape [N_frames, H, W, 3], values in [0,1]
    """
    assert flow_uv.dim() == 4 and flow_uv.size(3) == 2, 'Flow must have shape [N_frames, H, W, 2]'
    device = flow_uv.device

    if clip_flow is not None:
        flow_uv = torch.clamp(flow_uv, max=clip_flow)

    u = flow_uv[..., 0]
    v = flow_uv[..., 1]
    rad = torch.sqrt(u**2 + v**2)
    rad_max = rad.max()
    epsilon = 1e-10
    u = u / (rad_max + epsilon)
    v = v / (rad_max + epsilon)

    return flow_uv_to_colors(u, v, convert_to_bgr)

##########################################################################
# DOT args and helper class
##########################################################################

# dot_functions.py

dot_args = {
    "model": "dot",
    "fit_to": 336,
    "batch_size": 1,
    "num_tracks": 1024,
    "sim_tracks": 1024,
    "alpha_thresh": 0.8,
    "is_train": False,
    "worker_idx": 0,
    "num_workers": 2,
    "estimator_config": os.path.join(os.path.dirname(__file__), "configs/raft_patch_8.json"),
    "estimator_path": os.path.join(os.path.dirname(__file__), "checkpoints/cvo_raft_patch_8.pth"),
    "flow_mode": "direct",
    "refiner_config": os.path.join(os.path.dirname(__file__), "configs/raft_patch_4_alpha.json"),
    "refiner_path": os.path.join(os.path.dirname(__file__), "checkpoints/movi_f_raft_patch_4_alpha.pth"),
    "tracker_config": os.path.join(os.path.dirname(__file__), "configs/cotracker_patch_4_wind_8.json"),
    "tracker_path": os.path.join(os.path.dirname(__file__), "checkpoints/movi_f_cotracker_patch_4_wind_8.pth"),
    "sample_mode": "all",
    "interpolation_version": "torch3d",
    "inference_mode": "tracks_from_first_to_every_other_frame",
}

class Struct:
    def __init__(self, **kwargs):
        for key, value in kwargs.items():
            if isinstance(value, dict):
                self.__dict__[key] = Struct(**value)
            else:
                self.__dict__[key] = value

dot_args = Struct(**dot_args)

##########################################################################
# Function to create/reuse the DOT model
##########################################################################

def get_dot_model(width, height, dot_args, device=None):
    """
    Return the global DOT model if it exists and the width/height match;
    otherwise, create (or recreate) a new model with the specified size.
    """
    global _DOT_MODEL, _DOT_MODEL_WIDTH, _DOT_MODEL_HEIGHT

    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Check if the model is already loaded with the matching size
    if (_DOT_MODEL is not None
        and _DOT_MODEL_WIDTH == width
        and _DOT_MODEL_HEIGHT == height):
        return _DOT_MODEL

    # Otherwise, create or recreate the DOT model
    dot_args.width = width
    dot_args.height = height
    model = create_model(dot_args).to(device)
    model.eval()

    # Cache globally
    _DOT_MODEL = model
    _DOT_MODEL_WIDTH = width
    _DOT_MODEL_HEIGHT = height

    return _DOT_MODEL

##########################################################################
# Main function to extract the "dot frame" from a video (tensor)
##########################################################################

def get_dot_frame(video):
    """
    Given a video tensor of shape (F, C, H, W), runs DOT model inference 
    and returns a single flow_preview frame (the last one).
    """
    if video is None:
        raise ValueError("Failed to read video.")
    # video shape: (frames, channels, height, width)
    _f, _c, _h, _w = video.shape
    video = video.cuda()

    # Load or reuse the DOT model that matches _w x _h
    # (DOT can be sensitive to input dimension changes)
    dot_model = get_dot_model(_w, _h, dot_args, device=video.device)

    # Perform inference
    flow_pre = []
    with torch.no_grad():
        # The model expects shape [B, F, C, H, W], so add batch dim
        traj_pre = dot_model({"video": video.unsqueeze(0)}, mode=dot_args.inference_mode, **vars(dot_args))
        traj_pre = traj_pre["tracks"]  # e.g. shape [B, F, H, W, 5] (example shape)

        # Reorganize the tensor dimensions
        traj_pre = traj_pre.permute(0, 1, 4, 2, 3)[:, :, :2, ...]  # e.g. keep only first 2 channels (x/y)
        # Reshape to (F, 2, H, W)
        traj_pre = traj_pre.reshape(_f, 2, _h, _w)

        # Subtract the first frame to get relative flow
        first_frame = traj_pre[0].unsqueeze(0)  # shape (1, 2, H, W)
        traj_pre = traj_pre - first_frame  # broadcast subtract for each frame

        flow_pre.append(traj_pre)

    # Concatenate all frames along batch dimension
    flow_pre = torch.cat(flow_pre, dim=0)  # shape (F, 2, H, W)
    # Reorder to (F, H, W, 2)
    flow_pre = rearrange(flow_pre, 'n c h w -> n h w c')

    # Convert flow to a color visualization [F, H, W, 3]
    flow_preview = flow_to_color(flow_pre, clip_flow=None, convert_to_bgr=False)

    # Reorder back to (F, 3, H, W) for standard image usage
    flow_preview = rearrange(flow_preview, 'n h w c -> n c h w')

    # Return the last frame or a specific frame (e.g., flow_preview[-1])
    # If you specifically want the 24th frame, do flow_preview[23]
    return flow_preview[-1]
