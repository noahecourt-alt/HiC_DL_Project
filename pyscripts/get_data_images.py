import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

root = Path("../data_images")
root.mkdir(parents=True, exist_ok=True)

data = Path("../data")

names = []
ds = []

for dataset in sorted(data.iterdir()):
    names.append(dataset.name)
    hic = np.load(dataset / "hic.npy")
    U = np.load(dataset / "U.npy")
    ds.append((hic, U))

for i, (hmap, umap) in enumerate(ds):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10,5))
    hicimg = ax1.imshow(hmap, cmap="Reds", norm=LogNorm(vmin=1e-5, vmax=1))
    uimg = ax2.imshow(umap, cmap="RdBu", vmin=-0.4, vmax=0.4)
    ax1.set_title("Hi-C")
    ax2.set_title("U")
    fig.colorbar(hicimg, ax=ax1)
    fig.colorbar(uimg, ax=ax2)
    fig.tight_layout()
    fig.savefig(root / f"{names[i]}_data.png", dpi=200, bbox_inches="tight")
    plt.close(fig)
    
