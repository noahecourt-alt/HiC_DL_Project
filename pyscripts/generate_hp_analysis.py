from collections import defaultdict 
from pathlib import Path
import json
import matplotlib.pyplot as plt
from typing import Dict, Literal
import numpy as np
from loss_fns import get_loss
from scipy.stats import pearsonr
import matplotlib.image as img

def generate_hp_curves(root: Path, outpath: Path):
    all_dict = group_all(root)

    depth_list, depth_losses, depth_stds, _, _, depth_runtime, depth_runtime_stds = summary('depth', all_dict)
    lr_list, lr_losses, lr_stds, _, _, lr_runtime, lr_runtime_stds = summary('lr', all_dict)
    b_list, b_losses, b_stds, _, _, b_runtime, b_runtime_stds = summary('b', all_dict)
    bw_list, bw_losses, bw_stds, _, _, bw_runtime, bw_runtime_stds = summary('bw', all_dict)
    ks_list, ks_losses, ks_stds, _, _, ks_runtime, ks_runtime_stds = summary('ks', all_dict)

    fig, (ax1, ax2, ax3, ax4, ax5) = plt.subplots(5, 1, figsize=(10,10))

    ax1.errorbar(lr_list, lr_losses, yerr=lr_stds, color='r', label='Validation')
    ax1.set_title("Learning Rate")
    ax2.errorbar(depth_list, depth_losses, yerr=depth_stds, color='r')
    ax2.set_title("Depth")
    ax3.errorbar(b_list, b_losses, yerr=b_stds, color='r')
    ax3.set_title("Batch Size")
    ax4.errorbar(bw_list, bw_losses, yerr=bw_stds, color='r')
    ax4.set_title("Base Width")
    ax5.errorbar(
        range(len(ks_list)),
        ks_losses,
        yerr=ks_stds,
        color='r',
        fmt='o',
        linestyle='none',
        capsize=3,
    )
    ax5.set_xticks(range(len(ks_list)))
    ax5.set_xticklabels(ks_list)
    ax5.invert_xaxis()
    ax5.set_title("Kernel Sizes")
    for ax in (ax1, ax2, ax3, ax4, ax5):
        ax.set_ylabel("Mean Validation Loss")

    plt.tight_layout()

    plt.savefig(outpath / 'loss.png', dpi=200, bbox_inches='tight')
    plt.close(fig)

    fig, (ax1, ax2, ax3, ax4, ax5) = plt.subplots(5, 1, figsize=(10,10))

    ax1.errorbar(lr_list, lr_runtime, yerr=lr_runtime_stds, color='r', label='learning rate')
    ax1.set_title("Learning Rate")
    ax2.errorbar(depth_list, depth_runtime, yerr=depth_runtime_stds, color='r', label='unet depth')
    ax2.set_title("Depth")
    ax3.errorbar(b_list, b_runtime, yerr=b_runtime_stds, color='r', label='batch size')
    ax3.set_title("Batch Size")
    ax4.errorbar(bw_list, bw_runtime, yerr=bw_runtime_stds, color='r', label='base width')
    ax4.set_title("Base Width")
    ax5.errorbar(
        range(len(ks_list)),
        ks_runtime,
        yerr=ks_runtime_stds,
        color='r',
        fmt='o',
        linestyle='none',
        capsize=3,
    )
    ax5.set_title("Kernel Sizes")

    ax5.set_xticks(range(len(ks_list)))
    ax5.set_xticklabels(ks_list)
    ax5.invert_xaxis()

    for ax in (ax1, ax2, ax3, ax4, ax5): 
        ax.set_ylabel("Runtime (s)")

    plt.tight_layout()
    #fig.legend()
    plt.savefig(outpath / "runtime.png", dpi=200, bbox_inches='tight')
    plt.close(fig)

    parameters, p_vallosses, p_val_stds, _, _, p_runtimes, p_runtimes_stds = get_parameter_details(root)
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10,10))
    ax1.errorbar(parameters, p_vallosses, yerr=p_val_stds, color='r')
    ax2.errorbar(parameters, p_runtimes, yerr=p_runtimes_stds, color='r')
    ax1.set_ylabel('Validation Loss')
    ax1.set_title('Loss')
    ax2.set_ylabel('Runtime (s)')
    ax2.set_title('Runtime')
    ax1.set_xlabel("Number of Parameters")
    ax2.set_xlabel("Number of Parameters")

    plt.tight_layout()
    plt.savefig(outpath / "parameters.png", dpi=200, bbox_inches='tight')
    plt.close(fig)


def group_all(root: Path):
    all_hp = defaultdict(list)
    for path in root.glob('*.json'):
        rootstr = path.stem.split("_")
        with open(path) as f:
            file = json.load(f)
        all_hp[rootstr[1]].append(file)

    summary_stat = defaultdict(list) #to contain {'depth': ([valloss], [value]), 'lr': ([], []), etc. etc.}. valloss should be averaged across the 5 seeds

    return all_hp

def summary(ident: Literal['depth', 'b', 'bw', 'lr', 'ks', 'params'], all_dict: Dict): #summary function for each individual hyperparameter. all_dict will be the dictionary with all the hyperparameters in it. 

    if ident == 'depth':
        hp = 'depth'
    elif ident == 'b':
        hp = 'batch_size'
    elif ident == 'bw':
        hp = 'base_width'
    elif ident == 'lr':
        hp = 'learning_rate'
    elif ident == 'ks':
        hp = 'kernel_sizes'
    
    value_split_dict = defaultdict(list)
    value_list = []
    valloss_list = [] #list of the mean losses across the seeds for each individual value
    valstd_list = []
    testloss_list =[]
    teststd_list=[]
    runtime_list=[]
    runtime_std_list=[]

    for subdict in all_dict[ident]: #remember subdict is a dictionary within the all_dict. this loop is to split the larger dict into a dictionary that is split between values. so essentially the dictionary is depth, with each value being the different seeds
        if ident == 'ks':
            value = str(subdict[hp])
        else:
            value = subdict[hp]

        value_split_dict[value].append(subdict)

    for i in value_split_dict: #assuming this iterates through the keys. with i being the key
        vallosses = []
        testlosses=[]
        runtimes=[]
        value_list.append(i)
        for j in value_split_dict[i]: #iterating through the subdicts within the list held in the key
            vallosses.append(j['best_val_loss'])
            testlosses.append(j['test_loss'])
            runtimes.append(j['runtime'])
        val_mean_loss = np.mean(vallosses)
        test_mean_loss = np.mean(testlosses)
        val_std_loss = np.std(vallosses, ddof=1)
        test_std_loss = np.std(testlosses, ddof=1)
        runtime_mean = np.mean(runtimes)
        runtime_std = np.std(runtimes, ddof=1)
        valloss_list.append(val_mean_loss)
        valstd_list.append(val_std_loss)
        testloss_list.append(test_mean_loss)
        teststd_list.append(test_std_loss)
        runtime_list.append(runtime_mean)
        runtime_std_list.append(runtime_std)

    data_pairs = sorted(zip(value_list, valloss_list, valstd_list, testloss_list, teststd_list, runtime_list, runtime_std_list))
    value_list = [p[0] for p in data_pairs]
    valloss_list = [p[1] for p in data_pairs]
    valstd_list = [p[2] for p in data_pairs]
    testloss_list = [p[3] for p in data_pairs]
    teststd_list = [p[4] for p in data_pairs]
    runtime_list = [p[5] for p in data_pairs]
    runtime_std_list = [p[6] for p in data_pairs]

    return value_list, valloss_list, valstd_list, testloss_list, teststd_list, runtime_list, runtime_std_list


def get_parameter_details(root: Path):
    param_dict = defaultdict(list)

    for path in root.glob("*.json"):
        with open(path) as f:
            file = json.load(f)

        key = (file["depth"], file["base_width"], tuple(file["kernel_sizes"]))
        param_dict[key].append(file)

    params = []
    val_means = []
    val_stds = []
    test_means = []
    test_stds = []
    runtime_means = []
    runtime_stds = []

    for architecture, runs in param_dict.items():

        val_losses = [r["best_val_loss"] for r in runs]
        test_losses = [r["test_loss"] for r in runs]
        runtimes = [r["runtime"] for r in runs]
        num_params = runs[0]["num_parameters"]
        params.append(num_params)

        val_means.append(np.mean(val_losses))
        val_stds.append(np.std(val_losses, ddof=1) if len(val_losses) > 1 else 0.0)
        test_means.append(np.mean(test_losses))
        test_stds.append(np.std(test_losses, ddof=1) if len(test_losses) > 1 else 0.0)
        runtime_means.append(np.mean(runtimes))
        runtime_stds.append(np.std(runtimes, ddof=1) if len(runtimes) > 1 else 0.0)

    data_pairs = sorted(zip(params, val_means, val_stds, test_means, test_stds, runtime_means, runtime_stds))
    params = [p[0] for p in data_pairs]
    val_means = [p[1] for p in data_pairs]
    val_stds = [p[2] for p in data_pairs]
    test_means = [p[3] for p in data_pairs]
    test_stds = [p[4] for p in data_pairs]
    runtime_means = [p[5] for p in data_pairs]
    runtime_stds = [p[6] for p in data_pairs]

    return params, val_means, val_stds, test_means, test_stds, runtime_means, runtime_stds

def get_prob_loss(optim_path: Path, outpath: Path): #contact probability/strength plotted against loss
    #optim_path is the path to the outputs of the finalised, optimised model
    #loss is the loss function
    #post optimization. works on one test set
    hic = np.load(optim_path / 'reference_hic.npy').astype(np.float32)
    true_U = np.load(optim_path / 'trueU.npy').astype(np.float32)
    pred_U = np.load(optim_path / 'predictedU.npy').astype(np.float32)
    #loss = loss_fn(pred_U, true_U).item() BUT MUST BE PIXEL BY PIXEL!!
    losses = []
    contact_probs = []
    for i in range(hic.shape[0]):
        for j in range(hic.shape[1]):
            loss = abs(pred_U[i, j] - true_U[i, j]) / (abs(true_U[i, j]) + 1e-8)
            losses.append(loss)
            contact_probs.append(hic[i, j])
    return contact_probs, losses
    '''
    fig, ax = plt.subplots(figsize=(10, 10))
    ax.scatter(contact_probs, losses)
    ax.set_title('contact probability - losses')
    ax.set_ylabel('absrelloss')
    ax.set_xlabel('contact probabilities')

    #pearsonr(contact_probs, losses)

    plt.tight_layout()
    
    plt.savefig(outpath / "prob_loss.png", dpi=200, bbox_inches='tight')
    plt.close(fig)
    '''

def get_distance_loss(optim_path: Path, outpath: Path): #post optimization. works on one test set. 
    true_U = np.load(optim_path / 'trueU.npy').astype(np.float32)
    pred_U = np.load(optim_path / 'predictedU.npy').astype(np.float32)
    true_U = np.triu(true_U) #replaces lower triangle with 0s
    pred_U = np.triu(pred_U)
    losses = []
    distances = []
    for i in range(true_U.shape[0]):
        for j in range(true_U.shape[1]):
            if true_U[i, j] == 0:
                continue
            distances.append(abs(i - j))
            loss = abs(pred_U[i, j] - true_U[i, j]) / (abs(true_U[i, j]) + 1e-8)
            losses.append(loss)
    return distances, losses

    '''
    fig, ax = plt.subplots(figsize=(10, 10))
    ax.scatter(distances, losses)
    ax.set_title('distances - losses')
    ax.set_ylabel('absrelloss')
    ax.set_xlabel('distances')

    #pearsonr(contact_probs, losses)

    plt.tight_layout()
    
    plt.savefig(outpath / "distance_loss.png", dpi=200, bbox_inches='tight')
    plt.close(fig)
    '''

def generate_scatterplot(root: Path, outpath: Path, diagonal: bool = False): #post optimization. work on all the test sets
    fig, (ax1, ax2, ax3, ax4) = plt.subplots(4, 1, figsize=(10,14))
    colors = ['r', 'g', 'b', 'k', 'm', 'c', 'y', 'w']
    pearson_values = [] #for true_U vs pred_U plots
    slopes = []
    mae_vals, mse_vals, absrel_vals = [], [], []
    paths = []
    for i, run in enumerate(sorted(root.glob("*/*"))):
        contact_probs, c_losses = get_prob_loss(run, outpath)
        true_U = np.load(run / 'trueU.npy').flatten()
        pred_U = np.load(run / 'predictedU.npy').flatten()
        if diagonal:
            distances, d_losses = get_distance_loss(run, outpath)
            ax1.scatter(distances, d_losses, color=colors[i], alpha=0.1)
        loss = abs(pred_U - true_U) / (abs(true_U) + 1e-8)
        absrel_vals.append(np.mean(loss))

        r, _ = pearsonr(true_U, pred_U)
        pearson_values.append((colors[i], r))
        m, b = np.polyfit(true_U, pred_U, 1)
        slopes.append((colors[i], m))

        mae_vals.append(np.mean(np.abs(true_U - pred_U)))
        mse_vals.append(np.mean((true_U - pred_U)**2))

        ax2.scatter(contact_probs, c_losses, color=colors[i], alpha=0.1)
        ax3.scatter(true_U, pred_U, color=colors[i], alpha=0.1)
        xs = np.array([true_U.min(), true_U.max()])
        ax3.plot(xs, m*xs+b, color=colors[i], linewidth=2)
        paths.append(run)

    ax1.set_title("distance - loss")
    ax1.set_yscale('log')
    ax2.set_title("contact prob - loss")
    ax3.set_title("true - predicted")
    ax1.set_ylabel("absrelloss (log scale)")
    ax2.set_ylabel("absrelloss (log scale)")
    ax2.set_yscale('log')
    ax3.set_ylabel("predicted U")
    ax1.set_xlabel("distances (|i - j|)")
    ax2.set_xlabel("contact strengths")
    ax3.set_xlabel("true U")
    ax4.boxplot([mae_vals, mse_vals, absrel_vals], tick_labels=['MAE', 'MSE', 'AbsRelLoss'])
    ax4.set_yscale('log')  # <-- Fixes scale squashing
    ax4.set_title("Metric Distribution Across Runs (Log Scale)")
    ax4.set_ylabel("Value (Log Scale)")

    
    y = 0.01
    x = 0.05  

    for (color, r), (_, m) in zip(pearson_values, slopes):
        plt.figtext(x, y, f"r={r:.2f}, m={m:.2f}", color=color, fontsize=8, style='italic')
        x += 0.18 

    plt.tight_layout()
    fig.subplots_adjust(bottom=0.08)
    plt.savefig(outpath / "optim_summary_runtimeaccounted.png", dpi=200, bbox_inches='tight')
    plt.close(fig)

    return mae_vals, mse_vals, absrel_vals, paths

def generate_best_worst(root: Path, outpath: Path, num_pull, mae_vals, mse_vals, absrel_vals, paths): #for taking X number of best and worst predictions done by selected model. 
    
    mae, mse, ar, p = zip(*sorted(zip(mae_vals, mse_vals, absrel_vals, paths)))

    best = p[: num_pull]
    worst = p[-num_pull: ]    

    fig, axes = plt.subplots(2 * num_pull, 1, figsize=(8, 2 * num_pull * 6))
    axes = axes.flatten()

    axes[0].set_title("Good/Best predictions")

    for i, file in enumerate(best):
        image = img.imread(file / "comparison.png")
        axes[i].imshow(image)
        axes[i].axis('off')
        mae_value = mae[: num_pull][i]
        mse_value = mse[: num_pull][i]
        ar_value = ar[: num_pull][i]
        axes[i].text(0.5, -0.03, f"MAE: {mae_value:.4f}, MSE:{mse_value:.4f}, abrellos: {ar_value:.4f}", transform=axes[i].transAxes, ha="center", va="top", fontsize=10)
    

    axes[num_pull].set_title("Bad/Worst predictions")

    for i, file in enumerate(worst):
        image = img.imread(file / "comparison.png")
        axes[num_pull+i].imshow(image)
        axes[num_pull+i].axis('off')
        mae_value = mae[-num_pull: ][i]
        mse_value = mse[-num_pull: ][i]
        ar_value = ar[-num_pull: ][i]
        axes[num_pull+i].text(0.5, -0.03, f"MAE: {mae_value:.4f}, MSE:{mse_value:.4f}, abrellos: {ar_value:.4f}", transform=axes[num_pull+i].transAxes, ha="center", va="top", fontsize=10)

    plt.tight_layout()
    plt.savefig(outpath / "best_worst_pulls.png", dpi=200, bbox_inches='tight')




def main():
    #DATA_PATH = Path("../outputs_optim_2/results") #this is where all the outputs go from the runs
    DATA_PATH = Path("../final_runs_CNN/clip/run_final_diag_clip1_s45/test_predictions")
    #run_bw_b128_s45 was used for optim_summary.png
    #OUTPUT_PATH = Path("../outputs_optim_2/curves")
    OUTPUT_PATH = Path("../clip/sum_stats_diag_clip1")
    OUTPUT_PATH.mkdir(parents=True, exist_ok=True)
    #generate_hp_curves(DATA_PATH, OUTPUT_PATH)
    mae_vals, mse_vals, absrel_vals, paths = generate_scatterplot(DATA_PATH, OUTPUT_PATH, diagonal=True)
    generate_best_worst(DATA_PATH, OUTPUT_PATH, 2, mae_vals, mse_vals, absrel_vals, paths)

if __name__ == "__main__":
    main()
