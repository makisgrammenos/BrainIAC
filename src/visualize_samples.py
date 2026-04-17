import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
import yaml
import torch
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from adni_dataset import ADNIDataset, get_default_transform, get_validation_transform

def visualize_samples(dataset, n=5, title="", save_path="vis.png"):
    fig = plt.figure(figsize=(18, n * 4))
    fig.suptitle(title, fontsize=14)
    gs = gridspec.GridSpec(n, 3, figure=fig)

    for i in range(n):
        sample = dataset[i]
        img = sample['image'].squeeze(0).numpy()  # (D, H, W)
        label = int(sample['label'].item())

        d, h, w = img.shape
        slices = [img[d // 2], img[:, h // 2, :], img[:, :, w // 2]]
        slice_titles = ["Axial", "Coronal", "Sagittal"]

        for j, (sl, st) in enumerate(zip(slices, slice_titles)):
            ax = fig.add_subplot(gs[i, j])
            ax.imshow(sl, cmap='gray', origin='lower')
            ax.set_title(f"Sample {i} | Label={label} | {st}", fontsize=8)
            ax.axis('off')

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"Saved: {save_path}")
    plt.close()

if __name__ == "__main__":
    with open("config_finetune_amyl.yml", "r") as f:
        config = yaml.safe_load(f)

    image_size = tuple(config['data']['size'])
    root_dir = config['data']['root_dir']

    train_ds = ADNIDataset(
        csv_path=config['data']['csv_file'],
        root_dir=root_dir,
        transform=get_default_transform(image_size=image_size)
    )
    val_ds = ADNIDataset(
        csv_path=config['data']['val_csv'],
        root_dir=root_dir,
        transform=get_validation_transform(image_size=image_size)
    )

    visualize_samples(train_ds, n=5, title="Train set (with augmentations)", save_path="vis_train.png")
    visualize_samples(val_ds,   n=5, title="Val set (no augmentations)",     save_path="vis_val.png")
