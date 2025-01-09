#!/bin/bash

MOTION_DIR="/extra/xielab0/shanlins/code_base/videochat/data/italian_2nd/motion"
VID_DIR="/extra/xielab0/shanlins/code_base/videochat/data/italian_2nd/video"
TASK_NAME="italian_2nd"
LOG_DIR="/extra/xielab0/shanlins/code_base/videochat/data/italian_2nd/log"
GAP=288
DEVICE=0
TOTAL_TASKS=8

if [ -d "$LOG_DIR" ]; then
    rm -rf "$LOG_DIR"
fi

mkdir "$LOG_DIR"

cd /extra/xielab0/shanlins/code_base/videochat/Render
# ulimit -v 8388608

for ((TASK=0; TASK<TOTAL_TASKS; TASK++))
do
    START_MOTION=$((TASK * GAP))
    END_MOTION=$((TASK * GAP + GAP))

    if (( TASK %2 == 0 && TASK != 0 )); then
        DEVICE=$((DEVICE + 1))
    fi

    tmux new-session -d -s "render_task_${TASK}_${TASK_NAME}" bash -c "
        echo 'Starting rendering from $START_MOTION to $END_MOTION, using device $DEVICE' | tee -a '$LOG_DIR/render_log_${TASK}_${TASK_NAME}.txt';
        /extra/xielab0/haoyum3/blender-2.93.18-linux-x64/blender --background --python render2video.py -- --cfg=./configs/render.yaml --dir='$MOTION_DIR' --mode=video --device='$DEVICE' --task_name='$TASK_NAME' --start='$START_MOTION' --end='$END_MOTION' --save_dir='$VID_DIR' | tee -a '$LOG_DIR/render_log_${TASK}_${TASK_NAME}.txt';
        echo 'Finished rendering from $START_MOTION to $END_MOTION' | tee -a '$LOG_DIR/render_log_${TASK}_${TASK_NAME}.txt';
    "
done
