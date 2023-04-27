CUDA_VISIBLE_DEVICES="0,1"  python -W ignore train_DDP.py \
    --dataset sceneflow \
    --datapath /home/xp/Datas/Scene_Flow \
    --trainlist ./datalists/sceneflow_train.txt \
    --epochs 45 --lr 0.001  \
    --batch_size 8 \
    --maxdisp 192 \
    --model PSMNet \
    --loss_func  ADL \
    --savemodeldir /mnt/storage/xp/Check_Point/SceneFlow/   \
    --model_name  "SceneFlow_PSMNet_ADL"

# CUDA_VISIBLE_DEVICES="0" python -W ignore val.py \
#     --dataset sceneflow \
#     --datapath /home/xp/Datas/Scene_Flow \
#     --testlist ./datalists/sceneflow_test.txt \
#     --start_model 0 --end_model 44 --gap 1 \
#     --test_batch_size 1 \
#     --maxdisp 192 \
#     --model PSMNet \
#     --postprocess DM \
#     --model_name  "SceneFlow_PSMNet_ADL" \
#     --loadmodel /mnt/storage/xp/Check_Point/SceneFlow/SceneFlow_PSMNet_ADL
