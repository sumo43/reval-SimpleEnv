export MS2_ASSET_DIR=/home/xuanlin/Real2Sim/ManiSkill2_real2sim/data

env_name=MoveNearGoogleBakedTexInScene-v0
scene_name=google_pick_coke_can_1_v4
rgb_overlay_path=/home/xuanlin/Real2Sim/ManiSkill2_real2sim/data/real_impainting/google_move_near_real_eval_1.png
gpu_id=0

# declare -a arr=("/home/xuanlin/Real2Sim/rt_1_x_tf_trained_for_002272480_step/" \
#                 "/home/xuanlin/Real2Sim/xid77467904_000400120/" \
#                 "/home/xuanlin/Real2Sim/rt1poor_xid77467904_000058240/")
# declare -a arr=("/home/xuanlin/Real2Sim/xid77467904_000400120/")
# declare -a arr=("/home/xuanlin/Real2Sim/xid77467904_000400120/" \
#                 "/home/xuanlin/Real2Sim/rt_1_x_tf_trained_for_002272480_step/")
declare -a arr=("/home/xuanlin/Real2Sim/xid77467904_000400120/" \
                "/home/xuanlin/Real2Sim/rt_1_x_tf_trained_for_002272480_step/" \
                "/home/xuanlin/Real2Sim/rt1poor_xid77467904_000058240/" \
                "/home/xuanlin/Real2Sim/rt1new_77467904_000001120/")
for ckpt_path in "${arr[@]}"; do echo "$ckpt_path"; done

for ckpt_path in "${arr[@]}";

do CUDA_VISIBLE_DEVICES=${gpu_id} python real2sim/main_inference.py --policy-model rt1 --ckpt-path ${ckpt_path} \
  --robot google_robot_static \
  --control-freq 3 --sim-freq 513 --max-episode-steps 80 \
  --env-name ${env_name} --scene-name ${scene_name} \
  --rgb-overlay-path ${rgb_overlay_path} \
  --robot-init-x 0.35 0.35 1 --robot-init-y 0.21 0.21 1 --obj-variation-mode episode --obj-episode-range 0 60 \
  --robot-init-rot-quat-center 0 0 0 1 --robot-init-rot-rpy-range 0 0 1 0 0 1 -0.09 -0.09 1 \
  --additional-env-save-tags baked_7up_pepsi_color_adjust_redbull;

done





