import numpy as np
import random
import torch.nn as nn
import torch
import matplotlib.pyplot as plt

from pathlib import Path
import argparse

from torch.utils.data import DataLoader, random_split, Subset

from prep_fns import input_wrapper
from dataprep_fns import get_splits, Window_Wrapper, get_Dataloader, save_datasets, full_dataset
from loss_fns import get_loss 
from CNN_network import get_model

from collections import defaultdict
import json
import time

def test(model: nn.Module, data_set: DataLoader, loss: nn.Module, device):
    model.eval() #eval since we don't want it to update gradients or anything.
    sample_losses = []
    with torch.no_grad():
        for hic, U, _ in data_set: #hic, U has all the pairs within batch_idx
            hic = hic.to(device)
            U = U.to(device)
            Uhat = model(hic)
            for prediction, target in zip(Uhat, U):
                sample_losses.append(loss(prediction.unsqueeze(0), target.unsqueeze(0)).item())
    loss_mean = float(np.mean(sample_losses))
    loss_std = float(np.std(sample_losses, ddof=1)) if len(sample_losses) > 1 else 0.0 #ddof=1 for sample
    return loss_mean, loss_std

def save_matrix(root: Path, matrix, file_name: str): 
    root.mkdir(parents=True, exist_ok=True)
    np.save(root / file_name, matrix)

def print_settings(settings: dict, output_path: Path): #save settings of run in file
    with open(output_path / f"run_{settings['runnum']}" / f"settings.txt", 'w') as file:
        file.write(f"Run Number: {settings['runnum']}\nBatch Size: {settings['batchsize']}\nEigenvector Count (PCA): {settings['eigcount']}\nLearning Rate: {settings['learnrate']}\nEpochs: {settings['epochs']}\nWindow Size: {settings['windowsize']}\nKernel Sizes (In Order): {settings['kernelsizes']}")
        file.write(f"\nLoss: {settings['loss']}\nSave Best Model: {settings['save_model']}\nModel: {settings['model_name']}\nModel Depth: {settings['model_depth']}\nUNet Base Features: {settings['unet_base_features']}\nDevice: {settings['device']}\nDiagonal-Only Windows: {settings['diagonal']}\nSave Windows: {settings['save_windows']}\nSplit Mode: {settings['split_mode']}\nSplit Fractions (Test, Train, Validation): {settings['split']}\nSplit Seed: {settings['split_seed']}\nModel Seed: {settings['model_seed']}\nHoldout Sources: {settings['holdout']}\nInput Normalization: {settings['input_normalization']}\nWindow Stride Multiplier: {settings['window_stride_multiplier']}\nOptimizer: {settings['optimizer']}\nEarly-Stop Patience: {settings['early_stop_patience']}\nEarly-Stop Minimum Delta: {settings['early_stop_min_delta']}\nPotential Color Limit: {settings['potential_color_limit']}\nDifference Color Limit: {settings['difference_color_limit']}")

def print_losses(losses: list, output_path: Path, run: int, epochnum: int):
    with open(output_path / f"run_{run}" / f"losses.txt", 'a') as file:
        file.write(f"EPOCH {epochnum}\nepoch loss : {losses[0]}      validation loss: {losses[1]} ± {losses[2]}\n\n")

def save_matrix_png(root: Path, matrix, file_name: str, limit: float):
    fig, ax = plt.subplots(figsize=(6,5))
    image = ax.imshow(matrix, cmap="RdBu", vmin=-limit, vmax=limit)
    fig.colorbar(image, ax=ax)
    fig.tight_layout()
    fig.savefig(root / file_name, dpi=200)
    plt.close(fig)

def save_comparison_matrix(root: Path, prediction, target, file_name: str, potential_limit: float = 1.1, difference_limit: float = 1.1):
    difference = prediction - target
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    panels = ((prediction, "predicted U", potential_limit), (target, "true U", potential_limit), (difference, "difference", difference_limit))
    for ax, (matrix, name, limit) in zip(axes, panels):
        image = ax.imshow(matrix, cmap="RdBu", vmin=-limit, vmax=limit)
        ax.set_title(name)
        fig.colorbar(image, ax=ax)
    fig.tight_layout()
    fig.savefig(root / file_name, dpi=200)
    plt.close(fig)

def write_test_losses(root: Path, loss_value, loss_std, vloss_value, vloss_std, run_num):
    with open(root / "test_losses.txt", 'a') as file:
        file.write(f"run {run_num} test loss: {loss_value:0.6f} ± {loss_std:0.6f}     val loss: {vloss_value:0.6f} ± {vloss_std:0.6f} \n")

def main():
    start_time = time.time()

    parser = argparse.ArgumentParser()

    parser.add_argument("--run-id", required=True)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--eig-count", type=int, default=10)
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--window-size", type=int, default=192)
    parser.add_argument("--kernel-sizes", type=int, nargs="+", default=[3, 5, 7])
    parser.add_argument("--unet-depth", type=int, default=3)
    parser.add_argument("--unet-base-width", type=int, default=32)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--loss", choices=["absrelloss", "mse", "mselog"], default="absrelloss")
    parser.add_argument("--potential-color-limit", type=float, default=1.1)
    parser.add_argument("--difference-color-limit", type=float, default=1.0)
    parser.add_argument("--patience", type=int, default=25)
    parser.add_argument("--min-delta", type=float, default=1e-5)
    parser.add_argument("--diagonal", action="store_true")
    parser.add_argument("--save-windows", action="store_true")
    args = parser.parse_args()

    BATCH_SIZE = args.batch_size
    LEARNING_RATE = args.learning_rate
    EIG_COUNT = args.eig_count
    WINDOW_SIZE = args.window_size
    EPOCHS = args.epochs
    KERNEL_SIZES = args.kernel_sizes
    UNET_DEPTH = args.unet_depth
    UNET_BASE_WIDTH = args.unet_base_width
    RUN_NUM = args.run_id
    SAVE_MODEL = True
    MODEL_SEED = args.seed
    DIAGONAL = args.diagonal
    SAVE_WINDOWS = args.save_windows #save windows for GNN datasets

    random.seed(MODEL_SEED)
    np.random.seed(MODEL_SEED)
    torch.manual_seed(MODEL_SEED)
    torch.cuda.manual_seed_all(MODEL_SEED)

    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True)

    DATA_PATH = Path("../data") #run from the pyscripts folder

    DEVICE = "cuda"
    
    if SAVE_WINDOWS:
        DIAGONAL = True
    if DEVICE == "cuda":
        device = torch.device(DEVICE if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device("cpu")

    #OUTPUT_PATH = Path("../outputs") #this is where all the outputs go from the runs

    OUTPUT_PATH = Path("../final_runs_CNN") #this is where all the outputs go from the runs

    COMPLETE_SHUFFLE = False
    HOLDOUT=[]

    FORCE_SHUFFLE = True

    settings = {'runnum': RUN_NUM, 'batchsize': BATCH_SIZE, 'eigcount': EIG_COUNT, 'learnrate': LEARNING_RATE, 'epochs': EPOCHS, "windowsize": WINDOW_SIZE, "kernelsizes": KERNEL_SIZES}
    settings.update({'loss': args.loss, 'save_model': SAVE_MODEL, 'model_name': 'UNet_CNN', 'model_depth': UNET_DEPTH, 'unet_base_features': UNET_BASE_WIDTH, 'device': str(device), 'diagonal': DIAGONAL, 'save_windows': SAVE_WINDOWS, 'split_mode': 'force_shuffle' if FORCE_SHUFFLE else ('complete_shuffle' if COMPLETE_SHUFFLE else 'source_split'), 'split': [0.2, 0.6, 0.2], 'split_seed': 42, 'model_seed': MODEL_SEED, 'holdout': HOLDOUT, 'input_normalization': 'mean_fill', 'window_stride_multiplier': 1, 'optimizer': 'AdamW (weight_decay=1e-4)', 'early_stop_patience': args.patience, 'early_stop_min_delta': args.min_delta, 'potential_color_limit': args.potential_color_limit, 'difference_color_limit': args.difference_color_limit})
    RUN_PATH = OUTPUT_PATH / f"run_{RUN_NUM}"
    RUN_PATH.mkdir(parents=True, exist_ok=True)
    print_settings(settings, OUTPUT_PATH)

    generator = torch.Generator().manual_seed(42)
    split = [0.2, 0.6, 0.2] #for test train val

    if FORCE_SHUFFLE: #forcefully make splits that depict similar distributions. Use for smaller datasets
        test_list=[]
        train_list=[]
        val_list=[]
        base_dataset = full_dataset(DATA_PATH, lambda hic_map: input_wrapper(hic_map, eig=EIG_COUNT), HOLDOUT)
        all_windows = Window_Wrapper(base_dataset, blocksize=WINDOW_SIZE, stride=1, diagonal=DIAGONAL)

        windows_dict = defaultdict(list)
        for window_i, (source_i, r, c) in enumerate(all_windows.block_data):
            windows_dict[source_i].append(window_i) #stores in dictionary, source_i: window_i. could be source_i: [window_i, window_i]. basically, a dictionary and links the window # with source idx #; like windows 0, 1, 2 are from source 1 or smth

        for source_i, window_i in sorted(windows_dict.items()):
            n = len(window_i)
            testnum = int(n * split[0])
            if testnum < 1:
                testnum = 1
            trainnum = n - 2*testnum
            reorder = torch.randperm(n, generator=generator).tolist()
            test_list.extend(window_i[x] for x in reorder[0:testnum])
            train_list.extend(window_i[x] for x in reorder[testnum:testnum+trainnum])
            val_list.extend(window_i[x] for x in reorder[testnum+trainnum:])

        test_dataset = Subset(all_windows, test_list)
        train_dataset = Subset(all_windows, train_list)
        val_dataset = Subset(all_windows, val_list)

    elif COMPLETE_SHUFFLE:
        base_dataset = full_dataset(DATA_PATH, lambda hic_map: input_wrapper(hic_map, eig=EIG_COUNT), HOLDOUT)
        all_windows = Window_Wrapper(base_dataset, blocksize=WINDOW_SIZE, stride=1, diagonal=DIAGONAL)
        test_dataset, train_dataset, val_dataset = random_split(all_windows, split, generator)
    else:
        test_base, train_base, val_base, _, _, _ = get_splits(split, DATA_PATH, BATCH_SIZE, lambda hic_map: input_wrapper(hic_map, eig=EIG_COUNT), HOLDOUT, generator)
        for name, subset in [
            ("train", train_base),
            ("validation", val_base),
            ("test", test_base),
        ]:
            names = [subset.dataset.samples[i][0].parent.name for i in subset.indices] #subset.dataset refers to the original, complete dataset
            print(name, names)
        test_dataset = Window_Wrapper(test_base, blocksize=WINDOW_SIZE, stride=1, diagonal=DIAGONAL)
        train_dataset = Window_Wrapper(train_base, blocksize=WINDOW_SIZE, stride=1, diagonal=DIAGONAL)
        val_dataset = Window_Wrapper(val_base, blocksize=WINDOW_SIZE, stride=1, diagonal=DIAGONAL)
    if SAVE_WINDOWS:
        if FORCE_SHUFFLE or COMPLETE_SHUFFLE:
            save_datasets(Path("../GNNdata"), train_dataset, base_dataset, "train")
            save_datasets(Path("../GNNdata"), test_dataset, base_dataset, "test")
            save_datasets(Path("../GNNdata"), val_dataset, base_dataset, "validate")
        else:
            save_datasets(Path("../GNNdata"), train_dataset, train_base, "train")
            save_datasets(Path("../GNNdata"), test_dataset, test_base, "test")
            save_datasets(Path("../GNNdata"), val_dataset, val_base, "validate")

    test_dataloader = get_Dataloader(test_dataset, BATCH_SIZE, False)
    train_dataloader = get_Dataloader(train_dataset, BATCH_SIZE, True)
    val_dataloader = get_Dataloader(val_dataset, BATCH_SIZE, False)


    model = get_model("UNet_CNN", train_dataset[0][0].shape[0], kernel_sizes=KERNEL_SIZES, diagonal=DIAGONAL, depth=UNET_DEPTH, base_width=UNET_BASE_WIDTH).to(device) #number of channels in the transformed Hi-C input

    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=10, threshold=args.min_delta, threshold_mode='abs', min_lr=1e-6)

    loss = get_loss(args.loss)

    train_losses = []
    val_losses = []
    modelpath = OUTPUT_PATH / "models"
    modelpath.mkdir(parents=True, exist_ok=True)
    best_model_path = modelpath / f"CNNmodel_run_{RUN_NUM}"
    best_val_loss = float("inf")
    best_val_std = 0.0
    best_epoch = -1
    epochs_no_improvement = 0
    for epoch in range(EPOCHS):
        model.train()
        trainloss = 0.0
        sample_num = 0.0
        for hic, U, _ in train_dataloader:
            hic = hic.to(device)
            U = U.to(device)
            optimizer.zero_grad()
            pred = model(hic)
            loss_val = loss(pred, U)
            loss_val.backward()
            #grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            trainloss += loss_val.item() * hic.shape[0]
            sample_num += hic.shape[0]
        trainloss /= sample_num
        train_losses.append(trainloss)

        valloss, valstd = test(model, val_dataloader, loss, device)
        scheduler.step(valloss)
        if valloss < best_val_loss - args.min_delta:
            best_val_loss = valloss
            best_val_std = valstd
            best_epoch = epoch
            epochs_no_improvement = 0
            torch.save(model.state_dict(), best_model_path)
        else:
            epochs_no_improvement += 1
        val_losses.append(valloss)
        print_losses([trainloss, valloss, valstd], OUTPUT_PATH, RUN_NUM, epoch)
        if epochs_no_improvement >= args.patience:
            print(f"Stopped early at epoch {epoch}: no validation improvement for {args.patience} epochs.")
            break

    model.load_state_dict(
    torch.load(best_model_path, map_location=device, weights_only=True)
    )
    model.eval()
    
    fig_losses, ax = plt.subplots()
    ax.plot(train_losses, label = 'training losses', color = 'red')
    ax.plot(val_losses, label = 'validation losses', color = 'green')
    ax.set_xlabel("epoch")
    ax.set_ylabel(loss.get_name())

    testloss, teststd = test(model, test_dataloader, loss, device)
    runtime = time.time() - start_time
    results = {
        "run_id": RUN_NUM,
        "depth": UNET_DEPTH,
        "base_width": UNET_BASE_WIDTH,
        "learning_rate": LEARNING_RATE,
        "kernel_sizes": KERNEL_SIZES,
        "seed": MODEL_SEED,

        "best_epoch": best_epoch,
        "best_val_loss": best_val_loss,
        "best_val_std": best_val_std,

        "test_loss": testloss,
        "test_std": teststd,

        "eig_count": EIG_COUNT,
        "window_size": WINDOW_SIZE,
        "batch_size": BATCH_SIZE,
        "loss_name": args.loss,

        "runtime": runtime,
        "num_parameters": sum(p.numel() for p in model.parameters())
    }

    results_dir = OUTPUT_PATH / "results"
    results_dir.mkdir(exist_ok=True)

    with open(results_dir / f"run_{RUN_NUM}.json", "w") as file:
        json.dump(results, file, indent=4)

    write_test_losses(OUTPUT_PATH, testloss, teststd, best_val_loss, best_val_std, RUN_NUM)

    ax.errorbar(epoch, testloss, yerr=teststd, fmt='o', capsize=4, label='test loss')
    ax.legend()

    fig_losses.savefig(RUN_PATH / "loss_plot.png")

    plt.close(fig_losses)
    
    #fig, axes = plt.subplots(1, 2, figsize=(5, 5))
    print(
        f"Using best validation checkpoint from epoch {best_epoch} "
        f"(validation loss {best_val_loss:.6f})"
    )
    #bestloss = float("inf") #just setting the intial best loss to infinity
    
    test_outputs_root = RUN_PATH / "test_predictions"
    test_outputs_root.mkdir(parents=True, exist_ok=True)

    with torch.no_grad(): #this loop grabs the test set that performed the best
        for hic_batch, U_batch, wind_i in test_dataloader:
            batch_prediction = model(hic_batch.to(device))

            for x, wind_idx in enumerate(wind_i.tolist()):
                if FORCE_SHUFFLE or COMPLETE_SHUFFLE:
                    source_i, r, c = all_windows.block_data[wind_idx]
                    hic_path, _ = base_dataset.samples[source_i]
                else:
                    source_i, r, c = test_dataset.block_data[wind_idx] #source_i is the index in the original dataset. reminder that test dataset is now a window dataset)

                    #lossx = loss(prediction, target).item()
                    hic_path, _ = test_base.dataset.samples[source_i] #test_base is the pytorch subset, that points back to dataset
                sample_name = hic_path.parent.name

                prediction_matrix = batch_prediction[x, 0, :, :].cpu().numpy()
                target_matrix = U_batch[x, 0, :, :].cpu().numpy()

                block_size = target_matrix.shape[0] #returns the size of the window
                reference_hic = np.load(hic_path)[r: r + block_size, c: c + block_size] #np.load(hic_path) just loads the full original matrix, so need to crop it again

                sample_output = test_outputs_root / sample_name / f"r_{r}_{c}"
                sample_output.mkdir(parents=True, exist_ok=True)

                save_matrix(sample_output, prediction_matrix, "predictedU.npy")
                save_matrix(sample_output, target_matrix, "trueU.npy")
                save_matrix(sample_output, reference_hic, "reference_hic.npy")

                lmt = max(np.abs(prediction_matrix).max(), np.abs(target_matrix).max())

                save_matrix_png(sample_output, prediction_matrix, "predictedU.png", lmt)
                save_matrix_png(sample_output, target_matrix, "trueU.png", lmt)

                save_comparison_matrix(sample_output, prediction_matrix, target_matrix, "comparison.png", potential_limit=args.potential_color_limit, difference_limit=args.difference_color_limit)



                
                #if lossx < bestloss:
                #    bestloss = lossx
                #    best_prediction = prediction.squeeze().numpy()
                #    best_true = target.squeeze().numpy()
                #    best_idx = original_idx
                
    
    '''
    axes[0].imshow(best_prediction, cmap="RdBu", interpolation = None, vmin=-0.4, vmax=0.4)
    axes[0].set_title("predicted potential")
    axes[1].imshow(best_true, cmap="RdBu", interpolation = None, vmin=-0.4, vmax=0.4)
    axes[1].set_title("true potential")

    fig.savefig(RUN_PATH / "prediction_comparison.png", dpi=200, bbox_inches="tight")
    plt.close(fig)

    save_matrix(RUN_PATH, best_prediction, "predictedU.npy")
    '''

    #hic_path, u_path = test_dataset.dataset.samples[best_idx]
    #raw_hic = np.load(hic_path)
    #save_matrix(RUN_PATH, raw_hic, "reference_hic.npy")

if __name__ == "__main__":
    main()

'''
    model.eval()
    #bestloss = float("inf") #just setting the intial best loss to infinity

    test_outputs_root = RUN_PATH / "test_predictions"
    test_outputs_root.mkdir(parents=True, exist_ok=True)

    with torch.no_grad(): #this loop grabs the test set that performed the best
        for hic_batch, U_batch, original_indices in test_dataloader:
            batch_prediction = model(hic_batch)

            for x, original_idx in enumerate(original_indices):
                prediction = batch_prediction[x : x+1]
                target = U_batch[x : x+1]

                #lossx = loss(prediction, target).item()
                hic_path, _ = test_dataset.dataset.samples[original_idx]
                reference_hic = np.load(hic_path)

                sample_name = hic_path.parent.name
                sample_output = test_outputs_root / sample_name
                sample_output.mkdir(parents=True, exist_ok=True)

                prediction_matrix = prediction.squeeze().numpy()
                target_matrix = target.squeeze().numpy()

                save_matrix(sample_output, prediction_matrix, "predictedU.npy")
                save_matrix(sample_output, target_matrix, "trueU.npy")
                save_matrix(sample_output, reference_hic, "reference_hic.npy")

                lmt = max(np.abs(prediction_matrix).max(), np.abs(target_matrix).max())

                save_matrix_png(sample_output, prediction_matrix, "predictedU.png", lmt)
                save_matrix_png(sample_output, target_matrix, "trueU.png", lmt)
'''
