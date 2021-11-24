# 2D-gesture: window_size=64, sliding_step=64
# bptt 와 prediction window size 확인

# hidden : 16
python 1_train_predictor.py --data gesture \
                            --filename ann_gun_CentroidA.pkl \
                            --ckpt_name gesture_16 \
                            --nhid 16 \
                            --emsize 16 \
                            --nlayers 3 \
                            --lr 1e-3 \
                            --weight_decay 1e-06 \
                            --epochs 200 \
                            --batch_size 128 \
                            --eval_batch_size 128 \
                            --bptt 64 \
                            --prediction_window_size 64
                            

# hidden : 32
python 1_train_predictor.py --data gesture \
                            --filename ann_gun_CentroidA.pkl \
                            --ckpt_name gesture_32 \
                            --nhid 32 \
                            --emsize 32 \
                            --nlayers 3 \
                            --lr 1e-3 \
                            --weight_decay 1e-06 \
                            --epochs 200 \
                            --batch_size 128 \
                            --eval_batch_size 128 \
                            --bptt 64 \
                            --prediction_window_size 64

# hidden : 64
python 1_train_predictor.py --data gesture \
                            --filename ann_gun_CentroidA.pkl \
                            --ckpt_name gesture_64 \
                            --nhid 64 \
                            --emsize 64 \
                            --nlayers 3 \
                            --lr 1e-3 \
                            --weight_decay 1e-06 \
                            --epochs 200 \
                            --batch_size 128 \
                            --eval_batch_size 128 \
                            --bptt 64 \
                            --prediction_window_size 64
