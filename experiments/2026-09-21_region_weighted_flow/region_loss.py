"""Full-image flow MSE plus a height-normalized regional velocity MSE."""
import math

import torch
import torch.nn.functional as F


def region_flow_loss(prediction, target, mask, weight=1.0):
    if not math.isfinite(weight) or weight < 0:
        raise ValueError("weight must be finite/nonnegative")
    if prediction.shape != target.shape or prediction.ndim != 4:
        raise ValueError("Expected matching NCHW velocity tensors")
    if mask.shape != (prediction.shape[0], *prediction.shape[2:]):
        raise ValueError("OCR weights must match the velocity spatial grid")
    # The dataset validates masks on CPU. Compute reductions in fp32 under AMP.
    prediction, target = prediction.float(), target.float()
    mask = mask.to(device=prediction.device, dtype=torch.float32)
    area = mask.sum(dim=(1, 2))
    global_loss = F.mse_loss(prediction, target)
    # Cache w_i=min(1, h_ref/h_i) itself to preserve fp16 range, then square it
    # in the objective to obtain the intended inverse-height-squared correction.
    # The mean still includes every NCHW element, including zero-weight pixels.
    region_loss = ((prediction - target).square() * mask[:, None].square()).mean()
    loss = global_loss + weight * region_loss
    return loss, dict(flow_loss=global_loss, region_flow_loss=region_loss,
                      weighted_region_loss=weight * region_loss,
                      region_weight_mean=mask.mean(),
                      region_image_fraction=(area > 0).float().mean())
