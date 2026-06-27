"""TenCrop test-time augmentation."""
import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from .config import DatasetCfg


@torch.no_grad()
def evaluate_tencrop(
    model,
    test_dir: str,
    cfg: DatasetCfg,
    device: torch.device,
    batch_size: int = 32,
) -> float:
    """Average logits over 4 corner + 1 center + their flips (10 crops)."""
    model.eval()

    def to_tencrop(img):
        crops = transforms.TenCrop(cfg.crop_size)(
            transforms.Resize((cfg.img_size, cfg.img_size))(img)
        )
        norm = transforms.Normalize(cfg.norm_mean, cfg.norm_std)
        return torch.stack([norm(transforms.functional.to_tensor(c)) for c in crops])

    ds = datasets.ImageFolder(test_dir, transform=to_tencrop)
    # num_workers=0 because the closure over `cfg` is not picklable across procs.
    dl = DataLoader(ds, batch_size=batch_size, shuffle=False,
                    num_workers=0, pin_memory=True)

    correct, total = 0, 0
    for images, labels in dl:
        images = images.to(device, non_blocking=True)         # [B, 10, C, H, W]
        labels = labels.to(device, non_blocking=True)
        bs, ncrops, c, h, w = images.shape
        logits = model(images.view(-1, c, h, w))              # [B*10, K]
        logits = logits.view(bs, ncrops, -1).mean(1)          # avg over crops
        correct += (logits.argmax(1) == labels).sum().item()
        total   += labels.size(0)
    return correct / total
