#!/bin/bash

echo "Starting Android agent training script..."
echo "bash examples/acloo_trainer/online_android_train.sh"

set -x

if ! pgrep -f "ray" > /dev/null; then
    export RAY_PORT=2468
    export RAY_HEAD_IP=$(hostname -I | awk '{print $1}')
    ray start --head --port=$RAY_PORT --resources='{"docker:'$RAY_HEAD_IP'": 72}' &
    sleep 2
    echo y
fi

export PYTHONUNBUFFERED=1
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

ACTOR_MODEL_PATH=../../../models/UI-TARS-1.5-7B
PRM_MODEL_PATH=../../../models/UI-TARS-1.5-7B
TRAIN_TASK_FILES=data/train_instruct_examples.json
CRITIC_TRAIN_FILES=data/prm_data_examples/ac_prm_train_dataset_mini.csv

save_contents="['hf_model']"
NUM_GPUS=8
ROLLOUT_BSZ=8
SSMA=4
batch_size=$((16 * ROLLOUT_BSZ))
ppo_mini_batch_size=$((batch_size / 16 * NUM_GPUS))

experiment_name="uitars_7b_android_train_prm_SSMA${SSMA}_critic_prm"

python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=acloo \
    data.train_files=${TRAIN_TASK_FILES} \
    data.val_files=${TRAIN_TASK_FILES} \
    +data.critic_pretrain_files=${CRITIC_TRAIN_FILES} \
    data.train_batch_size=${batch_size} \
    data.critic_batch_size=${batch_size} \
    +data.pretrain_critic_batch_size=${batch_size} \
    data.max_prompt_length=32768 \
    data.max_response_length=512 \
    +data.max_pixels=1270180 \
    +data.min_pixels=256 \
    +data.rollout_batch_size=$ROLLOUT_BSZ \
    data.truncation='right' \
    data.shuffle=False \
    actor_rollout_ref.model.path=${ACTOR_MODEL_PATH} \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.checkpoint.save_contents="${save_contents}" \
    actor_rollout_ref.actor.strategy=fsdp2 \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.actor.fsdp_config.model_dtype=bf16 \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.ppo_mini_batch_size=${ppo_mini_batch_size} \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
    actor_rollout_ref.actor.use_kl_loss=False \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=4 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.6 \
    actor_rollout_ref.rollout.enable_chunked_prefill=False \
    actor_rollout_ref.rollout.n=${SSMA} \
    critic.enable=True \
    critic.optim.lr=1e-5 \
    critic.model.use_remove_padding=True \
    critic.optim.lr_warmup_steps_ratio=0.05 \
    critic.model.path=${PRM_MODEL_PATH} \
    critic.model.enable_gradient_checkpointing=True \
    critic.ppo_micro_batch_size_per_gpu=1 \
    critic.model.fsdp_config.param_offload=False \
    critic.model.fsdp_config.optimizer_offload=False \
    critic.model.fsdp_config.model_dtype=bf16 \
    critic.ppo_mini_batch_size=${ppo_mini_batch_size} \
    critic.use_prm=False \
    critic.rollout_n=1 \
    critic.pretrain_critic=True \
    +env.max_steps=16 \
    +env.num_envs=$ROLLOUT_BSZ \
    algorithm.use_kl_in_reward=False \
    trainer.val_before_train=False \
    trainer.critic_warmup=0 \
    trainer.logger='["console"]' \
    trainer.project_name='verl_acloo_uitars' \
    trainer.experiment_name=${experiment_name} \
    trainer.n_gpus_per_node=$NUM_GPUS \
    trainer.nnodes=1 \
    trainer.save_freq=12 \
    trainer.test_freq=1 \
    trainer.total_epochs=6 \
    2>&1 | tee -a logs/${experiment_name}.log