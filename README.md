**DESCRIPTION OVERVIEW of Python Scripts**


**MAIN COMPONENTS**

**CNN_network.py:**
Contains the U-Net CNN architecture.

**prep_fns.py:**
Conducts the feature extraction and the creation of the CNN input channels.

**dataprep_fns.py:**
Essentially prepares the datasets for the neural network.
1. Pulls the Hi-C contact maps along with their corresponding potential parameter maps.
2. Applies prep_fns.py channel creation to the datasets (before windows are made). 
3. Splits datasets into smaller windows.
4. Creates the stratified train/validate/test splits.
5. Applies Pytorch Dataloaders to the splits.

**train_test_val.py:**
The main training loop (including the validation and test).

**OTHER COMPONENTS**

**loss_fns.py:**
Define the loss functions used in the optimizer. 

**generate_hp_analysis.py:**
Generates summary statistics for a model run. This includes MAE, MSE, Pearson correlation, slope-regression analyses, loss - genomic distance distributions, and loss - contact strength distributions.

**run_sim.py:**
Calls **effective_simulation.py** to simulate the contact maps using the predicted potential parameter maps. 

Note: to use **run_sim.py** and **effective_simulation.py**, one needs to use chromdyn package (https://github.com/s-brahmachari/chromdyn). 
