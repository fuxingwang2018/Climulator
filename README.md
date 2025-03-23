# HCLIMAI
 Obtaining high-resolution data for convection-permitting climate models through deep learning

## Note on Atos

```
 module load python3/new
 mkdir -p $PERM/venvs
 cd $PERM/venvs
 python3 -m venv --system-site-packages climulator

 module load cuda
 source $PERM/venvs/climulator/bin/activate
 pip install mlxtend
 pip install keras


 module load python3/3.10.10-01
 cd $PERM/venvs
 python3 -m venv --system-site-packages climulator2
```

## Note on LUMI

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
