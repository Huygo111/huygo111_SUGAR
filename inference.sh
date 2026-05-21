TASK_NAME="$1"
TRACKER_CKPT="${2:-demo_ckpts/${TASK_NAME}/tracker.pt}"
GENERATOR_CKPT="${3:-demo_ckpts/${TASK_NAME}/generator.ckpt}"

python scripts/sugar_rl/play.py --task Sugar-G129dof-${TASK_NAME}-Inference \
    --checkpoint "${TRACKER_CKPT}" \
    --generator_checkpoint "${GENERATOR_CKPT}" \
    --motion_folder "data/${TASK_NAME}" \
    --num_envs 16 --eval_random_motion