# JAX DEPS
import simpler_env
from simpler_env.utils.env.observation_utils import get_image_from_maniskill2_obs_dict
import mediapy
import sapien.core as sapien
import os

# FASTAPI DEPS
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import StreamingResponse
from typing import List
import numpy as np
import cv2
import io
from simpler_env.policies.rt1.rt1_model import RT1Inference
import subprocess
import absl.logging
from uuid import uuid4
from time import sleep

import logging
logger = logging.getLogger(__name__)

absl.logging.set_verbosity(absl.logging.ERROR)

RT_1_CHECKPOINTS = {
    "rt_1_x": "rt_1_x_tf_trained_for_002272480_step",
    "rt_1_400k": "rt_1_tf_trained_for_000400120",
    "rt_1_58k": "rt_1_tf_trained_for_000058240",
    "rt_1_1k": "rt_1_tf_trained_for_000001120",
}

def run_cmd(cmd:str ):
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    tmp = proc.stdout.read()
    return tmp

def get_rt_1_checkpoint(name, ckpt_dir="./checkpoints"):
  assert name in RT_1_CHECKPOINTS, name
  ckpt_name = RT_1_CHECKPOINTS[name]
  ckpt_path = os.path.join(ckpt_dir, ckpt_name)
  print(ckpt_path)
  if not os.path.exists(ckpt_path):
    if name == "rt_1_x":
      command = ["gsutil", "-m", "cp", "-r", f"gs://gdm-robotics-open-x-embodiment/open_x_embodiment_and_rt_x_oss/{ckpt_name}.zip", f"{ckpt_dir}"]
      try:
        subprocess.run(command, check=True)
        print("command run")
      except subprocess.CalledProcessError as e:
        print(e.output)

      command = ["unzip", f"{ckpt_dir}/{ckpt_name}.zip", "-d", f"{ckpt_dir}"]

      try:
        subprocess.run(command, check=True)
        print("command run")
      except subprocess.CalledProcessError as e:
        print(e.output)
    else:
      run_cmd(f"gsutil -m cp -r gs://gdm-robotics-open-x-embodiment/open_x_embodiment_and_rt_x_oss/{ckpt_name} {ckpt_dir}")
  return ckpt_path

BASE_PATH="/workspace/videos"

app = FastAPI()

model_names = [
    "rt_1_x",
    "rt_1_400k",
    "rt_1_58k",
    "rt_1_1k",
    "octo_base",
    "octo_small"
]

policy_setup = "google_robot"


model_names = ["rt_1_x", "rt_1_400k", "rt_1_58k", "rt_1_1k"]#, "octo-base", "octo-small"]

ckpt_path = get_rt_1_checkpoint("rt_1_x")
rt_1_x_model = RT1Inference(saved_model_path=ckpt_path, policy_setup=policy_setup)
print("loaded rt_1_x")

ckpt_path = get_rt_1_checkpoint("rt_1_400k")
rt_1_400k_model = RT1Inference(saved_model_path=ckpt_path, policy_setup=policy_setup)
print("loaded rt_1_400k")

ckpt_path = get_rt_1_checkpoint("rt_1_58k")
rt_1_58k_model = RT1Inference(saved_model_path=ckpt_path, policy_setup=policy_setup)
print("loaded rt_1_58k")

ckpt_path = get_rt_1_checkpoint("rt_1_1k")
rt_1_1k_model = RT1Inference(saved_model_path=ckpt_path, policy_setup=policy_setup)
print("loaded rt_1_1k")

model_name_to_model = {
    "rt_1_x" : rt_1_x_model,
    "rt_1_400k": rt_1_400k_model,
    "rt_1_58k": rt_1_58k_model,
    "rt_1_1k": rt_1_1k_model
}

def setup_env(env_name: str, instruction: str):
    # @title Select your model and environment

    task_name = env_name
    
    #task_name = "google_robot_pick_coke_can"  # @param ["google_robot_pick_coke_can", "google_robot_move_near", "google_robot_open_drawer", "google_robot_close_drawer", "widowx_spoon_on_towel", "widowx_carrot_on_plate", "widowx_stack_cube", "widowx_put_eggplant_in_basket"]
    
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
    #instruction = env.get_language_instruction()

    if "google" in task_name:
      policy_setup = "google_robot"
    else:
      policy_setup = "widowx_bridge"
        
    return env, instruction #, policy_setup

def run_env(env, instruction, model_name):
    model = model_name_to_model[model_name]
    obs, reset_info = env.reset()
    #instruction = env.get_language_instruction()
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
    return images

def generate_video_path():
    return os.path.join(BASE_PATH, str(uuid4()) + ".mp4")

def arrays_to_video(arrays, output_path, fps=10):
    height, width, layers = arrays[0].shape
    size = (width, height)
    
    out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, size)
    
    for array in arrays:
        out.write(array)
    out.release()
    return 
    
def prompt2video(env_name: str, model_name: str, instruction_name:str):
    env, instruction = setup_env(env_name, instruction_name)
    images = run_env(env, instruction, model_name)
    video_path = generate_video_path()
    arrays_to_video(images, video_path)
    assert os.path.exists(video_path)
    return video_path

def merge_videos(video1_path, video2_path):
    merged_video_path = generate_video_path()

    command = f'ffmpeg -i {video1_path} -i {video2_path} -filter_complex "[0:v]pad=iw*2:ih[int];[int][1:v]overlay=W/2:0[vid]" -map "[vid]" -c:v libx264 -crf 23 -preset veryfast {merged_video_path}'
    #command = ["ffmpeg", "-i", video1_path, "-i", video2_path, "-filter_complex", 
    #'"[0:v]pad=iw*2:ih[int];[int][1:v]overlay=W/2:0[vid]"', "-map", "'[vid]'", "-c:v", "libx264", "-crf", "23", "-preset", "veryfast", merged_video_path]

    try:
        subprocess.run(command, check=True, shell=True)
        print("command run")
    except subprocess.CalledProcessError as e:
        print(e.output)
                                                                                                                                                                            
    return merged_video_path

@app.post("/create-video/")
async def create_video(env_name: str, model1_name:str, model2_name: str, instruction_name: str):
    video_path1 = prompt2video(env_name, model1_name, instruction_name)
    logger.info(f"saved video1 to {video_path1}")

    video_path2 = prompt2video(env_name, model2_name, instruction_name)
    logger.info(f"saved video2 to {video_path2}")


    merged_video_path = merge_videos(video_path1, video_path2)
    
    logger.info(f"saved video to {merged_video_path}")
    return StreamingResponse(io.open(merged_video_path, "rb"), media_type="video/mp4")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
