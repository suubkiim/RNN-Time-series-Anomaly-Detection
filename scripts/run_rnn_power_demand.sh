# Power-demand: : window_size=512, sliding_step=512
# bptt 와 prediction window size 확인

# hidden : 16
python 1_train_predictor.py --data power_demand \
                            --filename power_data.pkl \
                            --ckpt_name power_data_16 \
                            --nhid 16 \
                            --emsize 16 \
                            --nlayers 3 \
                            --lr 1e-3 \
                            --weight_decay 1e-06 \
                            --epochs 200 \
                            --batch_size 128 \
                            --eval_batch_size 128 \
                            --bptt 512 \
                            --prediction_window_size 512
                            

# hidden : 32
python 1_train_predictor.py --data power_demand \
                            --filename power_data.pkl \
                            --ckpt_name power_data_32 \
                            --nhid 32 \
                            --emsize 32 \
                            --nlayers 3 \
                            --lr 1e-3 \
                            --weight_decay 1e-06 \
                            --epochs 200 \
                            --batch_size 128 \
                            --eval_batch_size 128 \
                            --bptt 512 \
                            --prediction_window_size 512

# hidden : 64
python 1_train_predictor.py --data power_demand \
                            --filename power_data.pkl \
                            --ckpt_name power_data_64 \
                            --nhid 64 \
                            --emsize 64 \
                            --nlayers 3 \
                            --lr 1e-3 \
                            --weight_decay 1e-06 \
                            --epochs 200 \
                            --batch_size 128 \
                            --eval_batch_size 128 \
                            --bptt 512 \
                            --prediction_window_size 512
