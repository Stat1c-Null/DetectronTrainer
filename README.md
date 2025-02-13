1.Set up environment
  Create new conda environment
  conda create --name detectronenv python=3.11

2. Install Packages
  a.Install torch
  pip uninstall torch (optional in case torch needs to be reinstalled)
  pip cache purge (optional)
  pip install torch torchvision --pre -f https://download.pytorch.org/whl/nightly/cu121/torch_nightly.html
  pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu117
  https://discuss.pytorch.org/t/install-pytorch-with-cuda-12-1/174294/16

  b.Make sure torch is installed with Cuda enabled
  import torch
  print(torch.cuda.is_available())  # Should return True
  print(torch.version.cuda)

  c.pip install -r requirements.txt

  d.Make sure numpy is below version 2
  import numpy as np 
  print("NumPy version:", np.__version__)
  pip install --force-reinstall "numpy<2"

3.Download COCO Annotations, Training and Validation images
  https://cocodataset.org/#download
  Put everything in dataset folder

4.Clone detectron2 git project into the same folder(unless it gets installed with requirements.txt list)

5. Create configs folder
  Choose a vision model from model zoo: https://github.com/facebookresearch/detectron2/blob/main/MODEL_ZOO.md
  Download the config for that model by clicking on the name of the model. Place config inside of configs folder
  Many models have a base model, so you have to find a config for your base model. Read the config you have downloaded, and then find an appropriate config: https://github.com/facebookresearch/detectron2/tree/main/configs
  Download the model itself from model zoo and place it in the base folder

6. Run train.py file

