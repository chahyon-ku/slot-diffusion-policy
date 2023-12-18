# Slot Diffusion Policy

## Environement Setup

```bash
export SDP_ROOT=$PWD
mamba create -n sdp python=3.8
mamba activate sdp
mamba install pytorch==1.12.1 torchvision==0.13.1 torchaudio==0.12.1 cudatoolkit=11.3 -c pytorch
mamba install natsort pyquaternion absl-py zarr pandas diffusers numba pygame pymunk shapely scikit-image pyav threadpoolctl

cd $SDP_ROOT/slot_diffusion_policy/lib
# Install PerAct RLBench (https://github.com/peract/peract)
wget https://www.coppeliarobotics.com/files/V4_1_0/CoppeliaSim_Edu_V4_1_0_Ubuntu20_04.tar.xz
tar -xf CoppeliaSim_Edu_V4_1_0_Ubuntu20_04.tar.xz
git clone https://github.com/stepjam/PyRep.git
git clone -b peract https://github.com/MohitShridhar/RLBench.git
git clone -b peract https://github.com/MohitShridhar/YARR.git
pip install --no-deps -e ./PyRep/
pip install --no-deps -e ./RLBench
pip install --no-deps -e ./YARR

# DiffusionPolicy
git clone git@github.com:real-stanford/diffusion_policy.git
pip install --no-deps -e ./diffusion_policy
```

### Add to .bashrc
```bash
export COPPELIASIM_ROOT=$SLOT_DIFFUSION_POLICY_ROOT/slot_diffusion_policy/lib/CoppeliaSim_Edu_V4_1_0_Ubuntu20_04
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$COPPELIASIM_ROOT
export QT_QPA_PLATFORM_PLUGIN_PATH=$COPPELIASIM_ROOT
export DISPLAY=:0
```

## Dataset Generation
```bash
cd slot_diffusion_policy/lib/RLBench/tools
python dataset_generator.py --tasks=open_drawer \
                            --save_path=data/train \
                            --image_size=128,128 \
                            --renderer=opengl \
                            --episodes_per_task=100 \
                            --processes=1 \
                            --all_variations=True
```