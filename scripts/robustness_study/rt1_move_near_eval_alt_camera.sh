export MS2_ASSET_DIR=/home/xuanlin/Real2Sim/ManiSkill2_real2sim/data

declare -a env_arr=("MoveNearAltGoogleCameraInScene-v0" \
                   "MoveNearAltGoogleCamera2InScene-v0")
scene_name=google_pick_coke_can_1_v4
gpu_id=0

# "/home/xuanlin/Real2Sim/rt1poor_xid77467904_000058240/" 

declare -a arr=("/home/xuanlin/Real2Sim/rt_1_x_tf_trained_for_002272480_step/" \
                "/home/xuanlin/Real2Sim/xid77467904_000400120/" \
                "/home/xuanlin/Real2Sim/rt1new_77467904_000001120/" \
                "/home/xuanlin/Real2Sim/rt1poor_xid77467904_000058240/" \
                "/home/xuanlin/Real2Sim/gengap-phase1viscrop_b282197888_56381011_000368200/" \
                "/home/xuanlin/Real2Sim/gengap-phase1234viscrop_b284347745_56932176_000386120/")
for ckpt_path in "${arr[@]}"; do echo "$ckpt_path"; done

for env_name in "${env_arr[@]}";

do for ckpt_path in "${arr[@]}";

do CUDA_VISIBLE_DEVICES=${gpu_id} python real2sim/main_inference.py --policy-model rt1 --ckpt-path ${ckpt_path} \
  --robot google_robot_static \
  --control-freq 3 --sim-freq 513 --max-episode-steps 80 \
  --env-name ${env_name} --scene-name ${scene_name} \
  --robot-init-x 0.35 0.35 1 --robot-init-y 0.21 0.21 1 --obj-variation-mode episode --obj-episode-range 0 60 \
  --robot-init-rot-quat-center 0 0 0 1 --robot-init-rot-rpy-range 0 0 1 0 0 1 -0.09 -0.09 1;

done

done





