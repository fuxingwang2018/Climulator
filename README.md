# HCLIMAI
 Obtaining high-resolution data for convection-permitting climate models through deep learning

<details close>
<summary> $\textcolor{Blue}{\textsf{<u> Table of contents </u>}}$ </summary>
 
 1. [Installation and Run](#installation-and-run)

</details>

## Installation and Run

<details close>
<summary> $\textcolor{Blue}{\textsf{<u> Creating the environment </u>}}$ </summary>

### On ECMWF Atos

```
 module load python3/new
 mkdir -p $PERM/venvs
 cd $PERM/venvs
 python3 -m venv --system-site-packages climulator

 module load cuda
 source $PERM/venvs/climulator/bin/activate
 pip install mlxtend
 pip install keras

```

### On LUMI

```
 #module load cray-python/3.11.7
 module load cray-python/3.10.10
 mkdir -p $HOME/venvs
 cd $HOME/venvs
 python3 -m venv --system-site-packages Climulator
 source $HOME/venvs/Climulator/bin/activate
 pip install mlxtend  
 pip install keras==2.12.0  #tensorflow needs to be 2.12 to be consistent with keras
```

### On ALVIS/NAISS

```
 mkdir $HOME/venvs
 cd $HOME/venvs
 module load virtualenv/20.23.1-GCCcore-12.3.0
 virtualenv --system-site-packages climulator
 pip install mlxtend
 pip install "numpy<2" #Solution for error: A module that was compiled using NumPy 1.x cannot be run in NumPy 2.4.2 as it may crash.

```

### On local HPC (e.g., SMHI NSC Freja)
1. Install [Miniconda](https://docs.conda.io/en/latest/miniconda.html)

2. Activate conda in Miniconda first using 

    ```
    conda activate 
    ```
    or 
    ``` 
    source ~/miniconda3/bin/activate
    ```
3. Run the following command: 
    ``` 
    conda env create -f environment.yml

4. Make sure that you see “climulator” in the output when you ask for a
   list of all available environments:
    ``` 
    conda env list
    ```
5. Activating the 'climulator' environment:
    ```             
    conda activate climulator
    ```         
</details>

<details close>
<summary> $\textcolor{Blue}{\textsf{<u> How to run </u>}}$ </summary>
    
 1. Create a new folder to run experiments

    For example, on ATOS:
    ```
    mkdir -p <path-to-climulator>/run/ATOS/<name-of-the-new-folder>
    mkdir -p <path-to-climulator>/config/ATOS/<name-of-the-new-folder>
    ```
    (```path-to-climulator``` can be ```$HOME/Scripts/Climulator/``` where the repo is located)
    ```
    cd <path-to-climulator>

    cp config/ATOS/Test_Domain/config_main_SG_TestDomain_tas_atos.ini config/ATOS/<name-of-the-new-folder> <path-to-climulator>/

    cp run/ATOS/Test_Domain/run_main_SG_TD_tas_atos.sh run/ATOS/<name-of-the-new-folder> <path-to-climulator>/
    ```
    If needed, change file names of config_main_SG_TestDomain_tas_atos.ini  and run_main_SG_TD_tas_atos.sh.
 
 2. Configure settings in config_main.ini and run.sh
 
    In run.sh (e.g., run_main_SG_TD_tas_atos.sh), change the gpus/memory setup, and make sure the correct config_main.ini (e.g., config_main_SG_TestDomain_tas_atos.ini) is used.
 
    In config_main.ini (e.g., config_main_SG_TestDomain_tas_atos.ini), change the domain, date, path/file information in SETTINGS, and change the SRGAN model parameters in STATISTICS.

 3. Run
    ```
    cd <path-to-climulator>
    sbatch run/ATOS/run_main_SG_TD_tas_atos.sh
    ```

4. Outputs

    Outputs are set by path_main in config_main_*.ini. e.g., in config/ATOS/Test_Domain/config_main_SG_TestDomain_tas_atos.ini     
    ```
    path_main = '/ec/res4/scratch/smf/HCLIMAI/Test_Domain/SG/'
    ```
     
    The path to logs is set in run_*.sh. e.g., in     
    run/ATOS/Test_Domain/run_main_SG_TD_tas_atos.sh
    ```
    #SBATCH --chdir=/ec/res4/scratch/smf/log/log_ml
    ```

</details>


<details close>
<summary> $\textcolor{Blue}{\textsf{<u> Evaluation </u>}}$ </summary>

    Please refer to ClimulatorScore (https://github.com/fuxingwang2018/ClimulatorScore), which is developed to evaluate ML emulator outputs.

</details>
<p align="right">(<a href="#readme-top">back to top</a>)</p>





