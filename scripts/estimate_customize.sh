python train/train_customize.py --task_name DATE-train_mm1_albedo_eval_IIW_RESUME20230517-224958 \
    --if_train False --if_val True --if_vis True --eval_every_iter 4000 \
    --config-file train/configs/train_albedo.yaml --resume 20230517-224958--DATE-train_mm1_albedo \
    --data_root  /hdd/datasets/fipt/indoor_synthetic/kitchen/val/ldr_9/ \
    --dir_output /hdd/datasets/fipt/indoor_synthetic/kitchen/val/ldr_9/albedo