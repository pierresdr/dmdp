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




REM Bicycle

REM undelayed
taskset -ca 1-22 python run_trpo.py --env "Bicycle-v0" --mode train --seeds 1 --delay 0 --epochs 1000 --steps_per_epoch 5000 --max_ep_len 250 --delta 0.001 --v_hid 64 --v_l 1 --vf_lr 0.01 --v_iters 3 --pi_hid 64 --pi_l 2 --save_period 100 --save_dir "./output/bicycle/trpo_undelayed"
