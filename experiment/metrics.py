import torch
import torchvision.transforms as T
from PIL import Image, ImageDraw, ImageFont


def compute_lpips(img1, img2, net='alex'):
    """
    Compute LPIPS between two image tensors.

    Args:
        img1, img2: tensors of shape [1, 3, H, W] in [0, 1].
        net: LPIPS backbone ('alex' or 'vgg').
    Returns:
        float: LPIPS distance (lower = more similar).
    """
    import lpips
    loss_fn = lpips.LPIPS(net=net).to(img1.device)
    with torch.no_grad():
        return loss_fn(img1 * 2 - 1, img2 * 2 - 1).item()


def compute_ssim(img1, img2):
    """
    Compute SSIM between two image tensors.

    Args:
        img1, img2: tensors of shape [1, 3, H, W] in [0, 1].
    Returns:
        float: SSIM score (higher = more similar).
    """
    from torchmetrics.image import StructuralSimilarityIndexMeasure
    ssim = StructuralSimilarityIndexMeasure(data_range=1.0).to(img1.device)
    return ssim(img1, img2).item()


def compute_metrics(img1, img2):
    """Returns dict of all metrics between two image tensors."""
    return {
        "lpips": compute_lpips(img1, img2),
        "ssim": compute_ssim(img1, img2),
    }


def load_image_tensor(path, device='cuda'):
    """Load a PNG/JPG image as a [1, 3, H, W] tensor in [0, 1]."""
    img = Image.open(path).convert('RGB')
    tensor = T.ToTensor()(img).unsqueeze(0).to(device)
    return tensor


def make_comparison_grid(images_dict, nrow=None):
    """
    Create a labeled comparison grid from named images.

    Args:
        images_dict: {name: tensor} where tensor is [1, 3, H, W] in [0, 1].
        nrow: images per row (default: all in one row).
    Returns:
        PIL Image of the grid.
    """
    from torchvision.utils import make_grid

    names = list(images_dict.keys())
    tensors = [images_dict[n][0].cpu() for n in names]

    nrow = nrow or len(tensors)
    grid = make_grid(tensors, nrow=nrow, padding=4, pad_value=1.0)
    grid_pil = T.ToPILImage()(grid)

    # Add labels
    draw = ImageDraw.Draw(grid_pil)
    img_w = tensors[0].shape[2]
    padding = 4
    for i, name in enumerate(names):
        col = i % nrow
        row = i // nrow
        x = col * (img_w + padding) + padding
        y = row * (img_w + padding) + padding
        draw.text((x + 4, y + 4), name, fill=(255, 0, 0))

    return grid_pil
