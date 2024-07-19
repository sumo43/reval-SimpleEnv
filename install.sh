# for installation on a RUNPOD VM
# no sudo

curl https://packages.cloud.google.com/apt/doc/apt-key.gpg | apt-key add -
echo "deb https://packages.cloud.google.com/apt cloud-sdk main" | tee -a /etc/apt/sources.list.d/google-cloud-sdk.list
apt-get update && apt-get -yqq install google-cloud-cli
apt-get install -yqq --no-install-recommends vulkan-tools libnvidia-gl-535 ffmpeg vim tmux

# @title Install Real2Sim
git clone https://github.com/simpler-env/ManiSkill2_real2sim.git
pip install -e ./ManiSkill2_real2sim
git clone https://github.com/simpler-env/SimplerEnv.git
pip install -e ./SimplerEnv
mkdir ./SimplerEnv/checkpoints

pip install --quiet tf_agents

git clone https://github.com/octo-models/octo
cd ./octo && git checkout 653c54acde686fde619855f2eac0dd6edad7116b && cd ..
pip3 install -e ./octo

pip install --quiet mediapy
