MOTION_NPY="../test/motion/0.npy"
VID_DIR="../test/video"
TASK_NAME="FirstRound"
DEVICE=1

cd /extra/xielab0/haoyum3/Ask-Anything/videochat_finetue/Render
/extra/xielab0/haoyum3/blender-2.93.18-linux-x64/blender --background --python render2video.py -- --cfg=./configs/render.yaml --npy="$MOTION_NPY" --mode=video --device="$DEVICE" --task_name="$TASK_NAME" --save_dir="$VID_DIR"
