import os
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "chromdyn"))
import chromdyn
print("chromdyn path:", chromdyn.__file__)
from chromdyn.topology import TopologyGenerator
from chromdyn.chromatin_dynamics import ChromatinDynamics
from chromdyn.traj_utils import save_pdb

from chromdyn.hic_utils import HiCManager
hicman = HiCManager()

import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from IPython.display import HTML

from matplotlib.colors import LogNorm
from loss_fns import get_loss


def print_settings(settings: dict, output_path: Path): #save settings of run in file
    with open(output_path / f"settings.txt", 'w') as file:
        file.write(f"Run Number: {settings['runnum']}\nBatch Size: {settings['batchsize']}\nEigenvector Count (PCA): {settings['eigcount']}\nLearning Rate: {settings['learnrate']}\nEpochs: {settings['epochs']}")


def simulate_hic(loci: int, root: Path, platform="CPU", hic_batch_size=512, interpolymer=False, mass=1.0,
                 friction=1.0, kb=30.0, lb=1.0, kbound=30.0, Rc=10.0): #loci is the number of beads in the hic
    generator = TopologyGenerator()
    if interpolymer:
        chain_num = loci
        n_beads_per_chain = 1
    else:
        chain_num = 1 #intrapolymer interactions 
        n_beads_per_chain = loci

    N = n_beads_per_chain * chain_num # total number of beads

    chain_list = [n_beads_per_chain for i in range(chain_num)]
    type_list = []

    generator.gen_top(chain_lens = chain_list, types = "unique") #"unique" for FI effective maps N*N

    friction_phys = friction / mass

    sim = ChromatinDynamics(generator.topology, name='testEffective', platform_name=platform, output_dir=root, console_stream=True, mass=mass)

    #Ubonds
    #kb = 30.0
    #lb = 1.0
    sim.force_field_manager.add_harmonic_bonds(r0=lb, k = kb,)# this function will automatically add bonds according to the topology file

    #Usa
    #kbound = 30.0
    # note the formula of self avoidance here is different from OpenMiChroM
    sim.force_field_manager.add_self_avoidance(Ecut = 5.0, k = 20.0, r = 1.0) #r = sigma = 2*radius

    #Uboundary
    #Rc = 10.0
    sim.force_field_manager.add_flat_bottom_harmonic(k = kbound, r0 = Rc, center = (0.0, 0.0, 0.0))

    #Ueffective
    ff_matrix_path = root / "predictedU.npy"
    initial_ff_matrix = np.load(ff_matrix_path)
    interaction_matrix = initial_ff_matrix
    #interaction_matrix = -np.random.rand(generator.topology.getNumAtoms(), generator.topology.getNumAtoms()) # random interaction matrix for demonstration
    sim.force_field_manager.add_type_to_type_interaction(interaction_matrix = interaction_matrix,type_labels = generator.atom_types, mu=3.22, rc=1.78)

    block_size_prod = 500 # 0.1 tau_sim
    kT = 1.0
    #actually kT = 1 ~ T = 120.27, for simplicity, we set T = 120.0

    sim.simulation_setup(
        init_struct='randomwalk',
        integrator='langevin',
        temperature=kT*120.0,
        timestep=0.01,
        friction=friction_phys, # actually collision rate for any langevin type integrator in openmm
        save_pos=True,
        save_energy=True,
        energy_report_interval=5_000,
        pos_report_interval=block_size_prod,    
        PBC = False,
        box_vectors = (
            (10.0, 0.0, 0.0),
            (0.0, 10.0, 0.0),
            (0.0, 0.0, 10.0)
        ),   
    )

    #removed force_reinitialize = True
    relaxation_steps = 2000 * block_size_prod # 1000 tau_sim
    sim.run(relaxation_steps, report=False)
        
    #RUN PRODUCTION SIMULATION
    n_blocks_prod = 10000
    running_steps = n_blocks_prod * block_size_prod
    sim.run(running_steps, report=True)
    sim.print_force_info()

    sim.save_reports()
    save_pdb(sim) 
    # compute HiC
    hic = hicman.gen_hic_from_cndb(traj_file=sim.reporters.get('position').filename, mu=3.22, rc=1.78, p = None, platform=platform, parallel=True, batch_size=hic_batch_size)

    return hic

def compare_matrices(simulated_hic, reference_hic, root: Path, file_name: str): #mat2 is the true matrix, mat1 is the simulated matrix
    assert simulated_hic.shape == reference_hic.shape, "matrices must be of the same dimensions"

    difference = simulated_hic - reference_hic
    diff_lim = np.max(np.abs(difference))

    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15,5))

    reference_image = ax1.imshow(reference_hic, cmap="Reds", norm=LogNorm(vmin=1e-5, vmax=1))
    ax1.set_title("reference Hi-C")
    simulated_image = ax2.imshow(simulated_hic, cmap="Reds", norm=LogNorm(vmin=1e-5, vmax=1))
    ax2.set_title("simulated Hi-C")
    difference_image = ax3.imshow(difference, cmap="RdBu", vmin=-diff_lim, vmax=diff_lim)
    ax3.set_title("simulated - reference")

    fig.colorbar(reference_image, ax=ax1, shrink=0.8)
    fig.colorbar(simulated_image, ax=ax2, shrink=0.8)
    fig.colorbar(difference_image, ax=ax3, shrink=0.8)

    fig.savefig(root / file_name, dpi=200, bbox_inches="tight")
    plt.close(fig)

    loss = get_loss("absrelloss")

    simulated_tensor = torch.from_numpy(simulated_hic).float().unsqueeze(0).unsqueeze(0)
    reference_tensor = torch.from_numpy(reference_hic).float().unsqueeze(0).unsqueeze(0)
    hic_loss = loss(simulated_tensor, reference_tensor).item()
    return hic_loss
