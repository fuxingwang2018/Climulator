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

 #module load python3/3.10.10-01
 #cd $PERM/venvs
 #python3 -m venv --system-site-packages climulator2
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
    To not pollute the git repository, we suggest to create a new folder and copy the configuration file to the new folder.
    (example on ATOS)
    ```
    mkdir -p $HOME/<name-of-the-new-folder>
    cd $HOME/<name-of-the-new-folder>
    mkdir config
    mkdir -p run/ATOS
    cp <path-to-climulator>/config/config_main_SG_TestDomain_tas_atos.ini config/
    cp <path-to-climulator>/run/ATOS/run_main_SG_TD_tas_atos.sh run/ATOS/

    ```
 
 2. Configure settings in config_main.ini and run.sh
 
    to come ...
 
 3. Run
    ```
    cd $HOME/<name-of-the-new-folder>/run/ATOS
    sbatch run_main_SG_TD_tas_atos.sh 
    ```
 
</details>
<p align="right">(<a href="#readme-top">back to top</a>)</p>





