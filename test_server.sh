#!/bin/bash

# Variables
URL="http://127.0.0.1:8000/create-video/"
ENV_NAME="google_robot_pick_coke_can"
MODEL_NAME="rt_1_x"
OUTPUT_FILE="output.mp4"

# Make the POST request and save the video
wget "http://127.0.0.1:8000/create-video?env_name=$ENV_NAME&model_name=$MODEL_NAME" -O output.mp4 --method=POST

#echo "Video saved to $OUTPUT_FILE"

