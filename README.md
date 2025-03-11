# HCLIMAI
 Obtaining high-resolution data for convection-permitting climate models through deep learning

## Note on Atos

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

