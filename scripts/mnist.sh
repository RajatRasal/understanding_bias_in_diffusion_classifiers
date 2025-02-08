export RDMAV_FORK_SAFE=1
export OPENAI_LOG_FORMAT="stdout,log,csv,tensorboard"
export OPENAI_LOGDIR="./output/mnist"

poetry run train \
    --dataset mnist \
    --image_size 32 \
    --max_steps 100000 \
    --diffusion_steps 200 \
    --num_channels 64 \
    --num_res_blocks 1 \
    --attention_resolutions "-1" \
    --batch_size 128 \
    --channel_mult "1,4,8" \
    --use_fp16 True \
    --noise_schedule linear \
    --log_interval 100 \
    --save_interval 1000