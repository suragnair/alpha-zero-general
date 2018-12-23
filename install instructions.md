download git cmd
    https://git-scm.com/downloads

git clone https://github.com/JernejHabjan/alpha-zero-general.git

pip install 'tensorflow==1.8'

pip install numpy
pip install pygame


Tensorflow-Gpu installation:
    pip install 'tensorflow-gpu==1.8'
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
    - nvcc --version

    - make sure its added to env variables:
    ```
    - C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v9.0\bin
    - C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v9.0\libnvvp
    ```
    #### TensorFlow Gpu
    - Todo - check if i can only import whole conda environment from below tutorial
    - Install tensorflow in pycharm and not from conda cmd (launch it in admin)
    - Interpreter-> check "Use conda package manager button" on the right,
    - add package -> tensorflow-gpu, tensorflow, tensorboard, keras 2.13 - version is important!




    ### Graphviz and pydot:
    ```
    conda install graphviz
    conda install pydot
    ```
    Download Graphviz executable from [Here](https://graphviz.gitlab.io/_pages/Download/Download_windows.html)

    Add path to environmental variables and !important! restart Pycharm
    ```
    C:\Program Files (x86)\Graphviz2.38\bin
    ```



Setup pit and learn config:
    - alpha_zero_general/td2020/config.py -> CONFIG

For pit:
    - download release:
        https://github.com/JernejHabjan/alpha-zero-general/releases
    - extract
    - place extracted files in folder to
        alpha_zero_general/temp/
    - run alpha_zero_general/td2020/pit.py
for learn:


    - run alpha_zero_general/td2020/learn.py

### Ue4:
    download latest release https://github.com/JernejHabjan/TrumpDefense2020/releases
    run