REM Walker 2d
REM trpo
taskset -ca 78-88 python run_trpo.py --env "Walker2d-v2" --mode train --seeds 1 2 3 4 5 6 7 8 9 --delay 5 --epochs 1000 --steps_per_epoch 5000 --max_ep_len 250 --delta 0.001 --v_hid 64 --v_l 1 --vf_lr 0.01 --v_iters 3 --pi_hid 64 --pi_l 2 --save_period 100 --save_dir "./output/new_env/trpo_delay_5"
@REM dtrpo
taskset -ca 30-40 python run_dtrpo.py --env "Walker2d-v2" --seeds 3 4 5 6 7 8 9 --mode train --delay 5 --epochs 1000 --steps_per_epoch 5000 --max_ep_len 250 --delta 0.001 --v_hid 64 --v_l 1 --vf_lr 0.01 --v_iters 3 --pi_hid 64 --pi_l 2 --pretrain_epochs 25 --pretrain_steps 10000 --size_pred_buf 100000 --batch_size_pred 10000 --train_enc_iters 2 --enc_dim 64 --enc_causal --use_belief --enc_lr=0.01 --maf_lr=0.01 --save_period 100 --epochs_belief_training 200 --hidden_dim 16 --save_dir "./output/new_env/dtrpo_16"
@REM memoryless
taskset -ca 10-20 python run_trpo.py --mode train --env "Walker2d-v2" --memoryless --delay 5 --seeds 0 1 2 3 4 5 6 7 8 9 --epochs 1000 --steps_per_epoch 5000 --max_ep_len 250 --gamma 0.99 --delta 0.001 --v_hid 64 --v_l 1 --vf_lr 0.01 --v_iters 3 --pi_hid 64 --pi_l 2 --save_period 100 --save_dir "./output/new_env/memoryless"
@REM l2 trpo
taskset -ca 10-20 python run_dtrpo.py --env "Walker2d-v2" --mode train --seeds 0 1 2 3 4 5 6 7 8 9 --delay 5 --epochs 1000 --steps_per_epoch 5000 --max_ep_len 250 --delta 0.001 --v_hid 64 --v_l 1 --vf_lr 0.01 --v_iters 3 --pi_hid 64 --pi_l 2 --pretrain_epochs 2 --pretrain_steps 10000 --size_pred_buf 100000 --batch_size_pred 10000 --train_enc_iters 1 --enc_lr 0.005 --enc_dim 64 --enc_heads=2 --enc_l 1 --enc_ff 8 --enc_causal --enc_pred_to_pi --save_period=100 --save_dir "./output/new_env/l2trpo"




REM Pendulum
REM DTRPO Pendulum - 3 steps 
taskset -ca 34-44 python run_dtrpo_seeds.py --env "Pendulum-v0" --first_seed 6 --mode train --n_seeds 4 --delay 3 --epochs 2000 --steps_per_epoch 5000 --max_ep_len 250 --delta 0.001 --v_hid 64 --v_l 1 --vf_lr 0.01 --v_iters 3 --pi_hid 64 --pi_l 2 --pretrain_epochs 25 --pretrain_steps 10000 --size_pred_buf 100000 --batch_size_pred 10000 --train_enc_iters 2 --enc_dim 64 --enc_causal --use_belief --enc_lr=0.01 --maf_lr=0.01 --save_period 100 --epochs_belief_training 200 --hidden_dim 8 --save_dir "./output/dtrpo/delay_3_deter_belief"

REM DTRPO Pendulum - 5 steps 
taskset -ca 45-55 python run_dtrpo.py --env "Pendulum-v0" --mode train --seed 1 --delay 5 --epochs 1000 --steps_per_epoch 5000 --max_ep_len 250 --delta 0.001 --v_hid 64 --v_l 1 --vf_lr 0.01 --v_iters 3 --pi_hid 64 --pi_l 2 --pretrain_epochs 2 --pretrain_steps 10000 --size_pred_buf 100000 --batch_size_pred 10000 --train_enc_iters 2 --enc_dim 64 --enc_causal --use_belief --enc_lr=0.01 --maf_lr=0.01 --save_period 100 --epochs_belief_training 200 --hidden_dim 8
taskset -ca 57-67 python run_dtrpo_seeds.py --env "Pendulum-v0" --mode train --first_seed 3 --n_seeds 3 --delay 5 --epochs 2000 --steps_per_epoch 5000 --max_ep_len 250 --delta 0.001 --v_hid 64 --v_l 1 --vf_lr 0.01 --v_iters 3 --pi_hid 64 --pi_l 2 --pretrain_epochs 25 --pretrain_steps 10000 --size_pred_buf 100000 --batch_size_pred 10000 --train_enc_iters 2 --enc_dim 64 --enc_causal --use_belief --enc_lr=0.01 --maf_lr=0.01 --save_period 100 --epochs_belief_training 200 --hidden_dim 8 --save_dir "./output/dtrpo/delay_5_deter_belief"

REM DTRPO Pendulum - 10 steps 
taskset -ca 67-78 python run_dtrpo_seeds.py --env "Pendulum-v0" --mode train --first_seed 8 --n_seeds 2 --delay 10 --epochs 2000 --steps_per_epoch 5000 --max_ep_len 250 --delta 0.001 --v_hid 64 --v_l 1 --vf_lr 0.01 --v_iters 3 --pi_hid 64 --pi_l 2 --pretrain_epochs 25 --pretrain_steps 10000 --size_pred_buf 100000 --batch_size_pred 10000 --train_enc_iters 2 --enc_dim 64 --enc_causal --use_belief --enc_lr=0.01 --maf_lr=0.01 --save_period 100 --epochs_belief_training 200 --hidden_dim 8 --save_dir "./output/dtrpo/delay_10_deter_belief"

REM DTRPO Pendulum - 15 steps
taskset -ca 51-66 python run_dtrpo_seeds.py --env "Pendulum-v0" --mode train --first_seed 3 --n_seeds 7 --delay 15 --epochs 2000 --steps_per_epoch 5000 --max_ep_len 250 --delta 0.001 --v_hid 64 --v_l 1 --vf_lr 0.01 --v_iters 3 --pi_hid 64 --pi_l 2 --pretrain_epochs 25 --pretrain_steps 10000 --size_pred_buf 100000 --batch_size_pred 10000 --train_enc_iters 2 --enc_dim 64 --enc_causal --use_belief --enc_lr=0.01 --maf_lr=0.01 --save_period 100 --epochs_belief_training 200 --hidden_dim 8 --save_dir "./output/dtrpo/delay_15_deter_belief"

REM TRPO moutain car
taskset -ca 67-78 python run_trpo.py --env "MountainCarContinuous-v0" --mode train --seed 0 --delay 0 --epochs 500 --steps_per_epoch 5000 --max_ep_len 250 --delta 0.001 --v_hid 64 --v_l 1 --vf_lr 0.01 --v_iters 3 --pi_hid 64 --pi_l 2




REM Bicycle delay 5
REM undelayed
taskset -ca 12-22 python run_trpo.py --env "Bicycle-v0" --mode train --seeds 8 9 --delay 0 --epochs 500 --steps_per_epoch 5000 --max_ep_len 250 --delta 0.001 --v_hid 64 --v_l 1 --vf_lr 0.01 --v_iters 3 --pi_hid 64 --pi_l 2 --save_period 100 --save_dir "./output/bicycle/trpo_undelayed"
REM E-trpo
taskset -ca 78-88 python run_trpo.py --env "Bicycle-v0" --mode train --seeds 7 8 9 --delay 5 --epochs 500 --steps_per_epoch 5000 --max_ep_len 250 --delta 0.001 --v_hid 64 --v_l 1 --vf_lr 0.01 --v_iters 3 --pi_hid 64 --pi_l 2 --save_period 100 --save_dir "./output/bicycle/trpo_delay_5"
REM D-TRPO delay 5 hd 16
taskset -ca 1-10 python run_dtrpo.py --env "Bicycle-v0" --seeds 9 --mode train --delay 5 --epochs 500 --steps_per_epoch 5000 --max_ep_len 250 --delta 0.001 --v_hid 64 --v_l 1 --vf_lr 0.01 --v_iters 3 --pi_hid 64 --pi_l 2 --pretrain_epochs 25 --pretrain_steps 10000 --size_pred_buf 100000 --batch_size_pred 10000 --train_enc_iters 2 --enc_dim 64 --enc_causal --use_belief --enc_lr=0.01 --maf_lr=0.01 --save_period 100 --epochs_belief_training 200 --hidden_dim 16 --save_dir "./output/bicycle/dtrpo_dely_5_hd_16"
REM D-TRPO delay 5 hd 32
taskset -ca 1-22 python run_dtrpo.py --env "Bicycle-v0" --seeds 2 3 4 5 6 7 8 9 --mode train --delay 5 --epochs 500 --steps_per_epoch 5000 --max_ep_len 250 --delta 0.001 --v_hid 64 --v_l 1 --vf_lr 0.01 --v_iters 3 --pi_hid 64 --pi_l 2 --pretrain_epochs 25 --pretrain_steps 10000 --size_pred_buf 100000 --batch_size_pred 10000 --train_enc_iters 2 --enc_dim 64 --enc_causal --use_belief --enc_lr=0.01 --maf_lr=0.01 --save_period 100 --epochs_belief_training 200 --hidden_dim 32 --save_dir "./output/bicycle/dtrpo_dely_5_hd_32"
@REM l2 TRPO
taskset -ca 23-44 python run_dtrpo.py --env "Bicycle-v0" --mode train --seeds 3 4 5 6 7 --delay 5 --epochs 1000 --steps_per_epoch 5000 --max_ep_len 250 --delta 0.001 --v_hid 64 --v_l 1 --vf_lr 0.01 --v_iters 3 --pi_hid 64 --pi_l 2 --pretrain_epochs 2 --pretrain_steps 10000 --size_pred_buf 100000 --batch_size_pred 10000 --train_enc_iters 1 --enc_lr 0.005 --enc_dim 64 --enc_heads=2 --enc_l 1 --enc_ff 8 --enc_causal --enc_pred_to_pi --save_period=100 --save_dir "./output/bicycle/l2trpo_delay_5"
REM D-TRPO delay 5 hd 16 epochs_belief_training 100
taskset -ca 45-55 python run_dtrpo.py --env "Bicycle-v0" --seeds 9 --mode train --delay 5 --epochs 500 --steps_per_epoch 5000 --max_ep_len 250 --delta 0.001 --v_hid 64 --v_l 1 --vf_lr 0.01 --v_iters 3 --pi_hid 64 --pi_l 2 --pretrain_epochs 25 --pretrain_steps 10000 --size_pred_buf 100000 --batch_size_pred 10000 --train_enc_iters 2 --enc_dim 64 --enc_causal --use_belief --enc_lr=0.01 --maf_lr=0.01 --save_period 100 --epochs_belief_training 100 --hidden_dim 16 --save_dir "./output/bicycle/dtrpo_dely_5_hd_16_stop_100"
@REM memoryless
taskset -ca 45-55 python run_trpo.py --mode train --env "Bicycle-v0" --memoryless --delay 5 --seeds 0 1 2 3 4 5 6 7 8 9 --epochs 500 --steps_per_epoch 5000 --max_ep_len 250 --gamma 0.99 --delta 0.001 --v_hid 64 --v_l 1 --vf_lr 0.01 --v_iters 3 --pi_hid 64 --pi_l 2 --save_period 100 --save_dir "./output/new_env/m-trpo_delay_5"
REM D-TRPO delay 5 hd 32 epochs_belief_training 100
taskset -ca 12-22 python run_dtrpo.py --env "Bicycle-v0" --seeds 0 1 2 3 4 5 6 7 8 9 --mode train --delay 5 --epochs 500 --steps_per_epoch 5000 --max_ep_len 250 --delta 0.001 --v_hid 64 --v_l 1 --vf_lr 0.01 --v_iters 3 --pi_hid 64 --pi_l 2 --pretrain_epochs 25 --pretrain_steps 10000 --size_pred_buf 100000 --batch_size_pred 10000 --train_enc_iters 2 --enc_dim 64 --enc_causal --use_belief --enc_lr=0.01 --maf_lr=0.01 --save_period 100 --epochs_belief_training 100 --hidden_dim 32 --save_dir "./output/bicycle/d-trpo_delay_5_hd_32_stop_100"
REM D-TRPO delay 5 hd 8 epochs_belief_training 100
taskset -ca 1-22 python run_dtrpo.py --env "Bicycle-v0" --seeds 6 --mode train --delay 5 --epochs 500 --steps_per_epoch 5000 --max_ep_len 250 --delta 0.001 --v_hid 64 --v_l 1 --vf_lr 0.01 --v_iters 3 --pi_hid 64 --pi_l 2 --pretrain_epochs 25 --pretrain_steps 10000 --size_pred_buf 100000 --batch_size_pred 10000 --train_enc_iters 2 --enc_dim 64 --enc_causal --use_belief --enc_lr=0.01 --maf_lr=0.01 --save_period 100 --epochs_belief_training 100 --hidden_dim 8 --save_dir "./output/bicycle/d-trpo_delay_5_hd_8_stop_100"


REM BicycleRide delay 5
REM undelayed
taskset -ca 12-22 python run_trpo.py --env "BicycleRide-v0" --mode train --seeds 1 2 3 4 5 6 7 8 9 --delay 0 --epochs 500 --steps_per_epoch 5000 --max_ep_len 250 --delta 0.001 --v_hid 64 --v_l 1 --vf_lr 0.01 --v_iters 3 --pi_hid 64 --pi_l 2 --save_period 100 --save_dir "./output/bicycle_ride/trpo_delay_0"
REM E-trpo
taskset -ca 12-22 python run_trpo.py --env "BicycleRide-v0" --mode train --seeds  0 1 2 3 4 5 6 7 8 9 --delay 5 --epochs 500 --steps_per_epoch 5000 --max_ep_len 250 --delta 0.001 --v_hid 64 --v_l 1 --vf_lr 0.01 --v_iters 3 --pi_hid 64 --pi_l 2 --save_period 100 --save_dir "./output/bicycle_ride/trpo_delay_5"
REM D-TRPO delay 5 hd 16
taskset -ca 1-10 python run_dtrpo.py --env "BicycleRide-v0" --seeds  0 1 2 3 4 5 6 7 8 9 --mode train --delay 5 --epochs 500 --steps_per_epoch 5000 --max_ep_len 250 --delta 0.001 --v_hid 64 --v_l 1 --vf_lr 0.01 --v_iters 3 --pi_hid 64 --pi_l 2 --pretrain_epochs 25 --pretrain_steps 10000 --size_pred_buf 100000 --batch_size_pred 10000 --train_enc_iters 2 --enc_dim 64 --enc_causal --use_belief --enc_lr=0.01 --maf_lr=0.01 --save_period 100 --epochs_belief_training 200 --hidden_dim 16 --save_dir "./output/bicycle_ride/d-trpo_delay_5_hd_16"
@REM l2 TRPO
taskset -ca 67-88 python run_dtrpo.py --env "BicycleRide-v0" --mode train --seeds 0 1 2 3 4 5 6 7 8 9 --delay 5 --epochs 500 --steps_per_epoch 5000 --max_ep_len 250 --delta 0.001 --v_hid 64 --v_l 1 --vf_lr 0.01 --v_iters 3 --pi_hid 64 --pi_l 2 --pretrain_epochs 2 --pretrain_steps 10000 --size_pred_buf 100000 --batch_size_pred 10000 --train_enc_iters 1 --enc_lr 0.005 --enc_dim 64 --enc_heads=2 --enc_l 1 --enc_ff 8 --enc_causal --enc_pred_to_pi --save_period=100 --save_dir "./output/bicycle_ride/l2-trpo_delay_5"
@REM memoryless
taskset -ca 48-63 python run_trpo.py --mode train --env "BicycleRide-v0" --memoryless --delay 5 --seeds 0 1 2 3 4 5 6 7 8 9 --epochs 500 --steps_per_epoch 5000 --max_ep_len 250 --gamma 0.99 --delta 0.001 --v_hid 64 --v_l 1 --vf_lr 0.01 --v_iters 3 --pi_hid 64 --pi_l 2 --save_period 100 --save_dir "./output/bicycle_ride/m-trpo_delay_5"
REM D-TRPO delay 5 hd 8 epochs_belief_training 100
taskset -ca 1-22 python run_dtrpo.py --env "BicycleRide-v0" --seeds 0 1 2 3 4 5 6 7 8 9 --mode train --delay 5 --epochs 500 --steps_per_epoch 5000 --max_ep_len 250 --delta 0.001 --v_hid 64 --v_l 1 --vf_lr 0.01 --v_iters 3 --pi_hid 64 --pi_l 2 --pretrain_epochs 25 --pretrain_steps 10000 --size_pred_buf 100000 --batch_size_pred 10000 --train_enc_iters 2 --enc_dim 64 --enc_causal --use_belief --enc_lr=0.01 --maf_lr=0.01 --save_period 100 --epochs_belief_training 100 --hidden_dim 8 --save_dir "./output/bicycle_ride/d-trpo_delay_5_hd_8_stop_100"




@REM memory usage 
mprof run --multiprocess --include-children --python python run_dtrpo.py --mode "train" --env "Bicycle-v0" --delay "3" --delta "0.001" --pretrain_epochs "2" --pretrain_steps "250" --max_ep_len "250" --steps_per_epoch "250" --epochs "500" --enc_lr "0.005" --seeds "0" "1" --use_belief --enc_causal --batch_size_pred "250" 
mprof plot --flame
mprof plot --output memory-profile.png