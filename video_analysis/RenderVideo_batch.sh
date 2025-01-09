MOTION_DIR="/extra/xielab0/shanlins/code_base/videochat/data/le/motion"
SAVE_DIR="/extra/xielab0/shanlins/code_base/videochat/data/le/video"
TASK_NAME="le"
TASK=0
DEVICE=2
GAP=1

START_MOTION=$((TASK * GAP)) # inclusive
END_MOTION=$((TASK * GAP + GAP)) # noninclusive


# Navigate to the appropriate directory
cd /extra/xielab0/shanlins/code_base/videochat/Render


echo "Starting rendering from $START_MOTION to $END_MOTION" 
/extra/xielab0/haoyum3/blender-2.93.18-linux-x64/blender --background --python render2video.py -- --cfg=./configs/render.yaml --dir="$MOTION_DIR" --mode=video --device="$DEVICE" --task_name="$TASK_NAME" --start="$START_MOTION" --end="$END_MOTION" --save_dir="$SAVE_DIR" 
