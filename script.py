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
  return ckpt_path


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
    raw_action, action = model.step(image)
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
