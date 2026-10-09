envName=CNN_pytorch_2
mamba create --name $envName -y
mamba install -n $envName -y python=3.10 pytorch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 pytorch-cuda=12.1 matplotlib numpy pandas scikit-learn -c pytorch -c nvidia -c conda-forge
mamba run -n $envName python -m pip install chromdyn segmentation_models_pytorch IPython

