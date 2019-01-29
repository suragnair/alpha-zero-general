# RTS Game
*Jernej Habjan 2018*

This is a [diploma thesis project](https://github.com/JernejHabjan/Diploma-Thesis), which is an implementation of RTS game in Alpha Zero General wrapper created by Surag Nair in [suragnair/alpha-zero-general](https://github.com/suragnair/alpha-zero-general).
Game visualisation is also presented in PyGame and Unreal Engine 4.

## Requirements
- Recommended Python 3.6 (3.7 is not supported by TensorFlow by the time of writing(December 2018))
- Required TensorFlow (recommended 1.9)
- Optional Pygame (board outputs can be displayed also in console if Pygame is not used)
- Module can be connected via get_action.py to [UE4](https://github.com/JernejHabjan/TrumpDefense2020) 
## Files
Main files to start learning and pitting:
- rts/learn.py
- rts/pit.py
- rts/src/config_class.py

# Install instructions
download git cmd
> https://git-scm.com/downloads
open git bash cmd
```
git clone https://github.com/JernejHabjan/alpha-zero-general.git
```
run install script in 
> alpha-zero-general/rts/install.sh

## Tensorflow-Gpu installation (Optional):
```pip install 'tensorflow-gpu==1.8'```
### TensorFlow and CUDA
Install cuda:
- Install cuda files:
- cuda_9.0.176_win10
- cuda_9.0.176.1_windows
- cuda_9.0.176.2_windows
- cuda_9.0.176.3_windows
- Extract this folder and add it to Cuda path to corresponding folders:
    - cudnn-9.0-windows10-x64-v7.1

vertify cuda installation:
- ```nvcc --version```

- make sure its added to env variables:
```
- C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v9.0\bin
- C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v9.0\libnvvp
```

## Graphviz and pydot(Optional):
```
pip install graphviz
pip install pydot
```
Download Graphviz executable from [Here](https://graphviz.gitlab.io/_pages/Download/Download_windows.html)

Add path to environmental variables and !important! restart Pycharm
>C:\Program Files (x86)\Graphviz2.38\bin


# Running
## Setup pit and learn config:
- alpha_zero_general/rts/config.py -> CONFIG
## For pit:
- download release:
>https://github.com/JernejHabjan/alpha-zero-general/releases
- extract
- place extracted files in folder to
>alpha_zero_general/temp/
- and overwrite config file in
>alpha-zero-general/rts/src/config_class.py
- navigate to 
>C:\Users\USER\alpha-zero-general\rts
- run ```python pit.py```
## for learn:
- navigate to 
>C:\Users\USER\alpha-zero-general\rts
- run > ```python learn.py```
## Ue4:
    download latest release https://github.com/JernejHabjan/TrumpDefense2020/releases
    run
