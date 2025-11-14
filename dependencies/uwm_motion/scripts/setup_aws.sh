# Install conda
mkdir -p ~/miniconda3
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda3/miniconda.sh
bash ~/miniconda3/miniconda.sh -b -u -p ~/miniconda3
rm ~/miniconda3/miniconda.sh

# Refresh terminal
source ~/miniconda3/bin/activate

# Modify bashrc
cat << 'EOF' >> ~/.bashrc

# >>> conda initialize >>>
# !! Contents within this block are managed by 'conda init' !!
__conda_setup="$('/home/ubuntu/miniconda3/bin/conda' 'shell.bash' 'hook' 2> /dev/null)"
if [ $? -eq 0 ]; then
    eval "$__conda_setup"
else
    if [ -f "/home/ubuntu/miniconda3/etc/profile.d/conda.sh" ]; then
        . "/home/ubuntu/miniconda3/etc/profile.d/conda.sh"
    else
        export PATH="/home/ubuntu/miniconda3/bin:$PATH"
    fi
fi
unset __conda_setup
# <<< conda initialize <<<
EOF

# Create conda environment
conda create --yes -n val python==3.10.14
conda activate val

# Install requirements
pip install -r requirements.txt

# Clone robomimic fork
cd ..
git clone git@github.com:zchuning/robomimic.git
cd robomimic
pip install -e .

# Download robomimic dataset
python robomimic/scripts/setup_macros.py
python robomimic/scripts/download_datasets.py --tasks sim --dataset_types ph --hdf5_types raw
cd robomimic/scripts
source extract_obs_from_raw_datasets.sh

# Clone LIBERO fork
cd ..
git clone git@github.com:zchuning/LIBERO.git
cd LIBERO
pip install -e .
pip install -r requirements.txt

# Download LIBERO data
python benchmark_scripts/download_libero_datasets.py --datasets libero_100