# test
# prediction window

python 2_anomaly_detection.py --data gesture \
                              --filename ann_gun_CentroidA.pkl \
                              --ckpt_name gesture_16.pkl \
                              --prediction_window 64 \
                              --test_batch_size 128

python 2_anomaly_detection.py --data gesture \
                              --filename ann_gun_CentroidA.pkl \
                              --ckpt_name gesture_32.pkl \
                              --prediction_window 64 \
                              --test_batch_size 128

python 2_anomaly_detection.py --data gesture \
                              --filename ann_gun_CentroidA.pkl \
                              --ckpt_name gesture_64.pkl \
                              --prediction_window 64 \
                              --test_batch_size 128
