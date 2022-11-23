export BATCH_SIZE=1

export CUDA_VISIBLE_DEVICES=0,1
export CUDA_HOME=/usr/local/cuda-11.2
export TOKENIZERS_PARALLELISM=false
export MP_SIZE=1
export NUM_WORKERS=1
export NUM_GPUS_PER_WORKER=2

USE_TF=0 deepspeed --num_gpus=2 \
    ../../src/transformers/examples/pytorch/language-modeling/run_clm.py --output_dir=test --overwrite_output_dir --model_type=gpt2 --model_name_or_path=gpt2-xl --do_train --train_file=training-data.txt --per_device_train_batch_size $BATCH_SIZE --per_device_eval_batch_size $BATCH_SIZE --fp16 --deepspeed ds_config.json
