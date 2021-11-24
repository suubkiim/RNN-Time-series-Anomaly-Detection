# test
# prediction window

python 2_anomaly_detection.py --data power_demand \
                              --filename power_data.pkl \
                              --ckpt_name power_data_16.pkl \
                              --prediction_window 512

python 2_anomaly_detection.py --data power_demand \
                              --filename power_data.pkl \
                              --ckpt_name power_data_32.pkl \
                              --prediction_window 512

python 2_anomaly_detection.py --data power_demand \
                              --filename power_data.pkl \
                              --ckpt_name power_data_64.pkl \
                              --prediction_window 512