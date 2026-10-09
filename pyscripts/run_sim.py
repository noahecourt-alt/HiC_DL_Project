from pathlib import Path
import shutil
import numpy as np
from effective_simulation import simulate_hic, compare_matrices


def main():

    FILE_PATH = Path("../final_runs_CNN")
    #RUNNUM = 16
    #RUNNAME = f"run_{RUNNUM}"
    RUNNAME = "run_final_diag_s43"

    #sample_dir = FILE_PATH / RUNNAME / "test_predictions" / SAMPLE_NAME
    sample_dir = FILE_PATH / RUNNAME / "test_predictions" 

    SAMPLE_NAME = sorted(x.name for x in sample_dir.iterdir() if x.is_dir())

    for i, predicted_path in enumerate(sorted(sample_dir.glob("*/*/predictedU.npy"))):
        window_dir = predicted_path.parent
        reference_path = window_dir / "reference_hic.npy"

        run_dir = FILE_PATH / "simulation_outputs" / RUNNAME / SAMPLE_NAME[i]

        if not predicted_path.exists():
            raise FileNotFoundError(f"Missing predicted potential: {predicted_path}")
        if not reference_path.exists():
            raise FileNotFoundError(f"Missing reference Hi-C map: {reference_path}")

        predicted_u = np.load(predicted_path)
        reference_hic = np.load(reference_path)

        if predicted_u.ndim != 2 or predicted_u.shape[0] != predicted_u.shape[1]:
            raise ValueError(f"predictedU must be square; got {predicted_u.shape}")
        if not np.allclose(predicted_u, predicted_u.T):
            raise ValueError("predictedU must be symmetric.")

        if reference_hic.shape != predicted_u.shape:
            raise ValueError(
                f"Shape mismatch: predictedU is {predicted_u.shape}, "
                f"reference Hi-C is {reference_hic.shape}"
            )
        
        run_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy2(predicted_path, run_dir / "predictedU.npy")
        shutil.copy2(reference_path, run_dir / "reference_hic.npy")

        simulated_hic = simulate_hic(loci=predicted_u.shape[0], root=run_dir, platform="CUDA")
        np.save(run_dir / "simulated_hic.npy", simulated_hic)
        hic_loss = compare_matrices(simulated_hic=simulated_hic, reference_hic=reference_hic, root=run_dir, file_name="hic_comparison.png")

        with open(run_dir / "comparison_loss.txt", "w") as file:
            file.write(f"Absolute relative Hi-C loss: {hic_loss}\n")

        print(f"Saved simulation results to: {run_dir}")
        print(f"Hi-C loss: {hic_loss}")

if __name__ == "__main__":
    main()