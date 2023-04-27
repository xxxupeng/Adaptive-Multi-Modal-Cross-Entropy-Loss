CUDA_VISIBLE_DEVICES="0,1" python -W ignore train_DDP.py \
    --dataset kitti \
    --datapath /home/xp/Datas/KITTI_2015 \
    --trainlist ./datalists/kitti15_train.txt \
    --epochs 1000 --lr 0.001  \
    --batch_size 8 \
    --maxdisp 192 \
    --model PSMNet \
    --loss_func ADL \
    --savemodeldir /mnt/storage/xp/Check_Point/KITTI/   \
    --model_name  "KITTI15_PSMNet_ADL"

# CUDA_VISIBLE_DEVICES="0" python -W ignore val.py \
#     --dataset kitti \
#     --datapath /home/xp/Datas/KITTI_2015 \
#     --testlist ./datalists/kitti15_val.txt \
#     --start_model 19 --end_model 999 --gap 20 \
#     --test_batch_size 1 \
#     --maxdisp 192 \
#     --model PSMNet \
#     --postprocess DM \
#     --model_name  "KITTI15_PSMNet_ADL" \
#     --loadmodel /mnt/storage/xp/Check_Point/KITTI/KITTI15_PSMNet_ADL

