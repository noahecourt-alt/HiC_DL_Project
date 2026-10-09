import numpy as np
from torch.utils.data import Dataset, DataLoader, random_split, Subset
import torch
from pathlib import Path
from typing import Tuple


class full_dataset(Dataset):
    def __init__(self, root: Path, transform, holdout: list[str] | None = None):
        self.transform = transform
        self.samples = []

        if holdout is None:
            holdout = []

        for sample_dir in sorted(root.iterdir()):
            if sample_dir.name in holdout:
                continue
            hic_path = sample_dir / 'hic.npy'
            U_path = sample_dir / 'U.npy'

            assert hic_path.exists() and U_path.exists(), "hic or U path does not exist"

            self.samples.append((hic_path, U_path))

    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, i):
        hic_im = np.load(self.samples[i][0]).astype(np.float32)
        U_im = np.load(self.samples[i][1]).astype(np.float32)

        if hic_im.ndim != 2 or hic_im.shape != U_im.shape or hic_im.shape[0] != U_im.shape[1]:
            raise ValueError("Invalid paired hic and U maps")

        if self.transform is not None:
            hic_im = self.transform(hic_im)

        U_im = torch.tensor(U_im).unsqueeze(0)
        return hic_im, U_im, i

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, i):
        map_index, start = self.samples[i]
        end = start + self.window_size
        hic_map, U_map = self.maps[map_index]
        hic_im = hic_map[start:end, start:end].copy()
        U_im = U_map[start:end, start:end].copy()

        if self.transform is not None:
            hic_im = self.transform(hic_im)

        return hic_im, torch.tensor(U_im, dtype=torch.float32).unsqueeze(0)

def get_Dataloader(dataset: Dataset, batchsize: int, shuffle: bool):
    return DataLoader(dataset, batch_size=batchsize, shuffle=shuffle)

def get_splits(splits: Tuple[float, float, float], root: Path, batchsize: int, transform, holdout, generator):
    dataset = full_dataset(root=root, transform=transform, holdout=holdout)
    test_dataset, train_dataset, val_dataset = random_split(dataset, [splits[0], splits[1], splits[2]], generator) #don't really understand generators
    test_dataloader = get_Dataloader(test_dataset, batchsize, False)
    train_dataloader = get_Dataloader(train_dataset, batchsize, True)
    val_dataloader = get_Dataloader(val_dataset, batchsize, False)
    return test_dataset, train_dataset, val_dataset, test_dataloader, train_dataloader, val_dataloader 

class Window_Wrapper(Dataset): #used when wanting to cut windows into original matrices for more datasets
    def __init__(self, ds: Dataset, blocksize: int, stride: float, diagonal: bool=False):
        if stride <= 1:
            stride_val = int(blocksize * stride)
        else:
            stride_val = int(stride)
        self.hic_blocks = []
        self.U_blocks = []
        self.block_data = []

        if diagonal:
            for hic, U, idx in ds: #idx is the index of the hic and U in the original dataset
                H, W = hic.shape[-2:] #because hic has the shape of (channels, H, W)
                for r in range(0, H - blocksize + 1, stride_val):

                    block_hic = hic[:, r:r+blocksize, r:r+blocksize]
                    block_U = U[:, r:r+blocksize, r:r+blocksize]
        
                    self.hic_blocks.append(block_hic)
                    self.U_blocks.append(block_U)
                    self.block_data.append((idx, r, r))
        else:
            for hic, U, idx in ds: #idx is the index of the hic and U in the original dataset
                H, W = hic.shape[-2:] #because hic has the shape of (channels, H, W)
                for r in range(0, H - blocksize + 1, stride_val):
                    c_begin = r + blocksize + 4
                    for c in range(c_begin, W - blocksize + 1, stride_val):
                        #if c < r: #skip lower triangular blocks (since they are just a mirror of the upper triangular blocks)
                        #    continue

                        block_hic = hic[:, r:r+blocksize, c:c+blocksize]
                        block_U = U[:, r:r+blocksize, c:c+blocksize]
            
                        self.hic_blocks.append(block_hic)
                        self.U_blocks.append(block_U)
                        self.block_data.append((idx, r, c))
    def __len__(self):
        return len(self.hic_blocks)

    def __getitem__(self, i):
        return self.hic_blocks[i], self.U_blocks[i], i

def save_datasets(root: Path, window: Window_Wrapper, base_dataset, type_name: str): #root to be the general dataset path
    if type_name not in ["train", "validate", "test"]:
        raise ValueError("Type in save_datasets is not a valid type")
    for _, U, window_i in window:
        #source_i, r, c = window.block_data[i]
        #hic_path, U_path = base_dataset.dataset.samples[source_i] #base_dataset is a split of the main, full dataset, hence the usafe of .dataset

        window_source = window.dataset if isinstance(window, Subset) else window
        source_i, r, c = window_source.block_data[window_i]

        base_source = base_dataset.dataset if isinstance(base_dataset, Subset) else base_dataset
        hic_path, U_path = base_source.samples[source_i]


        sample_name = hic_path.parent.name
        block_size = U.shape[-1]
        raw_hic = np.load(hic_path)[r: r+block_size, c: c+block_size]
        raw_U = np.load(U_path)[r: r+block_size, c: c+block_size]

        wind_dir = root / type_name / f"{sample_name}_r{r}_c{c}"
        wind_dir.mkdir(parents=True, exist_ok=True)

        np.save(wind_dir / "hic.npy", raw_hic)
        np.save(wind_dir / "U.npy", raw_U)
