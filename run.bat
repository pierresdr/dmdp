REM DTRPO Pendulum 
taskset -ca 45-55 python run_dtrpo.py --env "Pendulum-v0" --mode train --seed 0 --delay 5 --epochs 1000 --steps_per_epoch 5000 --max_ep_len 250 --delta 0.001 --v_hid 64 --v_l 1 --vf_lr 0.01 --v_iters 3 --pi_hid 64 --pi_l 2 --pretrain_epochs 2 --pretrain_steps 10000 --size_pred_buf 100000 --batch_size_pred 10000 --train_enc_iters 1 --enc_lr 0.005 --enc_dim 64 --enc_causal --use_belief --enc_lr=0.001 --maf_lr=0.001 --save_period 100 --epochs_belief_training 200 --hidden_dim 4

REM TRPO moutain car
taskset -ca 65-75 python run_trpo.py --env "MountainCarContinuous-v0" --mode train --seed 0 --delay 0 --epochs 1000 --steps_per_epoch 5000 --max_ep_len 250 --delta 0.001 --v_hid 64 --v_l 1 --vf_lr 0.01 --v_iters 3 --pi_hid 64 --pi_l 2  