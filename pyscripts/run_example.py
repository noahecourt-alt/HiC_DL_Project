import torch
from pathlib import Path
from CNN_network import get_model
from rebuildCNN import save_comparison_matrix
from prep_fns import input_wrapper
import numpy as np
from loss_fns import get_loss

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

depth = 3
diagonal = False

model_num = 15
model_path = Path(f'../model_file/CNNmodel_run_{model_num}')

model = get_model("UNet_CNN", in_channels=14, kernel_sizes=[3, 5, 7], diagonal=diagonal).to(device) #number of channels in the transformed Hi-C input

saved_weights = torch.load(model_path, map_location=device, weights_only=True)

model.load_state_dict(saved_weights)

model.eval()

#root = Path('../example_files/sample_ecoli_10kb')
root = Path('../data/sample_2_10')

hic_path = root / 'hic.npy'
U_path = root / 'U.npy'

raw_hic = np.load(hic_path).astype(np.float32)
U = np.load(U_path).astype(np.float32)

divisible = 2**depth

n = raw_hic.shape[0]
remove = n % divisible
raw_hic = raw_hic[0: n - remove, 0: n - remove]
U = U[0: n - remove, 0: n - remove]

input_hic = input_wrapper(raw_hic, eig=10).to(device)
input_hic = input_hic.unsqueeze(0) #shape [1, 14, height, width]

with torch.no_grad():
    prediction = model(input_hic)

predicted_U = prediction[0, 0].cpu().numpy()

true_U = torch.from_numpy(U).unsqueeze(0).unsqueeze(0).to(device)


loss_fn = get_loss("absrelloss")
loss = loss_fn(prediction, true_U)

print(loss.item())

example_outputs = Path(f'../example_output')
example_outputs.mkdir(parents=True, exist_ok=True)
save_comparison_matrix(
    example_outputs,
    predicted_U,
    U,
    f"model_{model_num}_comparison.png",
    potential_limit=1.1,
    difference_limit=0.3,
)
