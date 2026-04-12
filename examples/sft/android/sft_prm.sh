#!/bin/bash
set -x

if [ "$#" -lt 2 ]; then
    echo "Usage: examples/sft/android/sft_prm.sh <nproc_per_node> <save_path> [other_configs...]"
    exit 1
fi

nproc_per_node=$1
save_path=$2
train_batch_size=16
# Shift the arguments so $@ refers to the rest
shift 2

torchrun --nnodes=1 --nproc_per_node=$nproc_per_node \
     -m verl.trainer.prm_sft_trainer \
    data.train_files=data/prm_data_examples/ac_prm_train_dataset_mini.csv \
    data.val_files=data/prm_data_examples/ac_prm_train_dataset_mini.csv \
    data.multiturn.enable=true \
    data.multiturn.messages_key=messages \
    data.micro_batch_size_per_gpu=2 \
    data.train_batch_size=$train_batch_size \
    data.max_length=8192 \
    +data.model_path=../../../models/UI-TARS-1.5-7B \
    model.partial_pretrain=../../../models/UI-TARS-1.5-7B \
    model.strategy=fsdp2 \
    model.fsdp_config.cpu_offload=False \
    model.fsdp_config.offload_params=False \
    +model.freeze_vision_tower=False \
    +optim.strategy=adamw_bf16 \
    trainer.default_local_dir=$save_path \
    trainer.project_name=multiturn-sft \
    trainer.experiment_name=multiturn-sft-uitars-1.5-7B-sp2 \
    trainer.logger=console \
    trainer.total_epochs=3 \
    trainer.checkpoint.save_contents=['hf_model'] \
    ulysses_sequence_parallel_size=1 \
    use_remove_padding=true \
    # trainer.save_freq=702 \