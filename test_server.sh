#!/bin/bash

# Variables
URL="http://127.0.0.1:8000/create-video/"
ENV_NAME="google_robot_pick_coke_can"
MODEL_NAME="rt_1_x"
INSTRUCTION_NAME="pick up coke can"
OUTPUT_FILE="pickup.mp4"

# Make the POST request and save the video
wget "http://127.0.0.1:8000/create-video?env_name=$ENV_NAME&model_name=$MODEL_NAME&instruction_name=$INSTRUCTION_NAME" -O $OUTPUT_FILE --method=POST

#echo "Video saved to $OUTPUT_FILE"

URL="http://127.0.0.1:8000/create-video/"
ENV_NAME="google_robot_pick_coke_can"
MODEL_NAME="rt_1_x"
INSTRUCTION_NAME="move coke can to the right side of the table"
OUTPUT_FILE="tap.mp4"

# Make the POST request and save the video
wget "http://127.0.0.1:8000/create-video?env_name=$ENV_NAME&model_name=$MODEL_NAME&instruction_name=$INSTRUCTION_NAME" -O $OUTPUT_FILE --method=POST


