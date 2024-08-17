#!/usr/bin/env python
# coding: utf-8

# <a href="https://colab.research.google.com/github/simpler-env/SimplerEnv/blob/main/example.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

# # SimplerEnv: Simulated Manipulation Policy Evaluation for Real-World Robots
# 
# - Project page: <https://simpler-env.github.io/>
# - Code: <https://github.com/simpler-env/SimplerEnv>

# ## Installation
# 

# In[1]:
import os


#@title [!Important]Please use a GPU runtime.
#get_ipython().system('nvidia-smi')


# In[2]:


# @title Install vulkan for rendering
#get_ipython().system('apt-get update')
#get_ipython().system('apt-get install -yqq --no-install-recommends vulkan-tools libnvidia-gl-535 ffmpeg vim tmux')
#get_ipython().system('pip3 install scipy==1.12.0')
# below fixes some bugs introduced by some recent Colab changes
#!mkdir -p /usr/share/vulkan/icd.d
#!wget -q -P /usr/share/vulkan/icd.d https://raw.githubusercontent.com/haosulab/ManiSkill/main/docker/nvidia_icd.json
#!wget -q -O /usr/share/glvnd/egl_vendor.d/10_nvidia.json https://raw.githubusercontent.com/haosulab/ManiSkill/main/docker/10_nvidia.json


# In[3]:


# @title Make sure vulkan is installed correctly
#get_ipython().system('vulkaninfo | head -n 5')


# In[4]:


# @title Install Real2Sim
#get_ipython().system('git clone https://github.com/simpler-env/ManiSkill2_real2sim.git')
#get_ipython().system('pip install -e ./ManiSkill2_real2sim')
#get_ipython().system('git clone https://github.com/simpler-env/SimplerEnv.git')
#get_ipython().system('pip install -e ./SimplerEnv')
#get_ipython().system('mkdir ./SimplerEnv/checkpoints')


# In[5]:


#@title [Optional]Install RT-1 dependencies
#get_ipython().system(' pip install --quiet tf_agents')


# In[ ]:


#@title [Optional]Install Octo dependencies
#get_ipython().system('git clone https://github.com/octo-models/octo')
#get_ipython().system('cd ./octo && git checkout 653c54acde686fde619855f2eac0dd6edad7116b && cd ..')
#get_ipython().system('pip3 install -e ./octo')
#get_ipython().system('pip3 install distrax==0.1.5 "einops>= 0.6.1"')
#get_ipython().system('pip3 install flax')
#get_ipython().system('pip3 install "tensorflow==2.15"')
#get_ipython().system('pip3 install orbax-checkpoint==0.4.4')
#get_ipython().system('pip3 install tensorflow transformers')
#get_ipython().system('pip3 install --upgrade "jax[cuda12_pip]==0.4.20"  "jaxlib[cuda12_pip]==0.4.20" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html --no-deps')


# In[8]:


# @title Install other requirements
#get_ipython().system('pip install --quiet mediapy')


# In[1]:


# @title [Important]Post Installation

# run this so local pip installs are recognized
import site
site.main()


# ## Create a Simulated Environment and Take Random Actions

# In[3]:


import simpler_env
from simpler_env.utils.env.observation_utils import get_image_from_maniskill2_obs_dict
import mediapy
import sapien.core as sapien

task_name = "google_robot_pick_coke_can"  # @param ["google_robot_pick_coke_can", "google_robot_move_near", "google_robot_open_drawer", "google_robot_close_drawer", "widowx_spoon_on_towel", "widowx_carrot_on_plate", "widowx_stack_cube", "widowx_put_eggplant_in_basket"]

if 'env' in locals():
  print("Closing existing env")
  env.close()
  del env
env = simpler_env.make(task_name)
# Colab GPU does not supoort denoiser
sapien.render_config.rt_use_denoiser = False
obs, reset_info = env.reset()
instruction = env.get_language_instruction()
print("Reset info", reset_info)
print("Instruction", instruction)

frames = []
done, truncated = False, False
while not (done or truncated):
   # action[:3]: delta xyz; action[3:6]: delta rotation in axis-angle representation;
   # action[6:7]: gripper (the meaning of open / close depends on robot URDF)
   image = get_image_from_maniskill2_obs_dict(env, obs)
   action = env.action_space.sample() # replace this with your policy inference
   obs, reward, done, truncated, info = env.step(action)
   frames.append(image)

episode_stats = info.get('episode_stats', {})
print("Episode stats", episode_stats)
mediapy.show_video(frames, fps=10)


# In[ ]:


#get_ipython().system('pip3 install sapien==')


# ## Run Inference on Simulated Environments

# In[4]:


# @title Setup

import os
import numpy as np
import simpler_env
from simpler_env.utils.env.observation_utils import get_image_from_maniskill2_obs_dict
import mediapy


RT_1_CHECKPOINTS = {
    "rt_1_x": "rt_1_x_tf_trained_for_002272480_step",
    "rt_1_400k": "rt_1_tf_trained_for_000400120",
    "rt_1_58k": "rt_1_tf_trained_for_000058240",
    "rt_1_1k": "rt_1_tf_trained_for_000001120",
}


def get_rt_1_checkpoint(name, ckpt_dir="./SimplerEnv/checkpoints"):
  assert name in RT_1_CHECKPOINTS, name
  ckpt_name = RT_1_CHECKPOINTS[name]
  ckpt_path = os.path.join(ckpt_dir, ckpt_name)
  if not os.path.exists(ckpt_path):
    if name == "rt_1_x":
      os.system('gsutil -m cp -r gs://gdm-robotics-open-x-embodiment/open_x_embodiment_and_rt_x_oss/{ckpt_name}.zip {ckpt_dir}')
      os.system('unzip {ckpt_dir}/{ckpt_name}.zip -d {ckpt_dir}')
    else:
      os.system('gsutil -m cp -r gs://gdm-robotics-open-x-embodiment/open_x_embodiment_and_rt_x_oss/{ckpt_name} {ckpt_dir}')
  return ckpt_path


# In[5]:


# @title Select your model and environment

task_name = "google_robot_pick_coke_can"  # @param ["google_robot_pick_coke_can", "google_robot_move_near", "google_robot_open_drawer", "google_robot_close_drawer", "widowx_spoon_on_towel", "widowx_carrot_on_plate", "widowx_stack_cube", "widowx_put_eggplant_in_basket"]

if 'env' in locals():
  print("Closing existing env")
  env.close()
  del env
env = simpler_env.make(task_name)

# Note: we turned off the denoiser as the colab kernel will crash if it's turned on
# To use the denoiser, please git clone our SIMPLER environments
# and perform evaluations locally.
sapien.render_config.rt_use_denoiser = False

obs, reset_info = env.reset()
instruction = env.get_language_instruction()
print("Reset info", reset_info)
print("Instruction", instruction)

if "google" in task_name:
  policy_setup = "google_robot"
else:
  policy_setup = "widowx_bridge"


# In[8]:


#get_ipython().system('pip3 install orbax-checkpoint==0.4.4')


# In[9]:


# @title Select your model and environment
from simpler_env.policies.openvla.openvla_model import OpenVLAInference

#model_name = "octo-base" # @param ["rt_1_x", "rt_1_400k", "rt_1_58k", "rt_1_1k", "octo-base", "octo-small"]

model = OpenVLAInference(policy_setup=policy_setup)


# In[10]:


#@title Run inference

obs, reset_info = env.reset()
instruction = env.get_language_instruction()
model.reset(instruction)
print(instruction)

image = get_image_from_maniskill2_obs_dict(env, obs)  # np.ndarray of shape (H, W, 3), uint8
images = [image]
predicted_terminated, success, truncated = False, False, False
timestep = 0
while not (predicted_terminated or truncated):
    # step the model; "raw_action" is raw model action output; "action" is the processed action to be sent into maniskill env
    raw_action, action = model.step(image, instruction)
    predicted_terminated = bool(action["terminate_episode"][0] > 0)
    obs, reward, success, truncated, info = env.step(
        np.concatenate([action["world_vector"], action["rot_axangle"], action["gripper"]])
    )
    print(timestep, info)
    # update image observation
    image = get_image_from_maniskill2_obs_dict(env, obs)
    images.append(image)
    timestep += 1

episode_stats = info.get("episode_stats", {})
print(f"Episode success: {success}")


# In[11]:


#print(task_name, model_name)
#mediapy.show_video(images, fps=10)


# ## Gallery

# In[ ]:


# @markdown RT-1-X close drawer
#print(task_name, model_name)
#mediapy.show_video(images, fps=10)
# Note: we turned off the denoiser as the colab kernel will crash if it's turned on
# To use the denoiser, please git clone our SIMPLER environments
# and perform evaluations locally.


# In[ ]:


# @markdown Octo-base widowx_put_eggplant_in_basket
#print(task_name, model_name)
#mediapy.show_video(images, fps=10)

