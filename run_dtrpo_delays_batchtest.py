import os

os.system("conda activate Thesis")

seeds = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
delays = [0.7, 0.6, 0.55]
models = ['DET-Delay10']

# DTRPO Deterministic Delay Pendulum Tests
for model in models:
    for seed in seeds:
        save_dir = "./output/dtrpo/Pendulum-AISTATS/Results-" + model \
                   + "/Pendulum-AISTATS-run" + str(seed)

        for delay in delays:
            print('[METHOD]: DTRPO\t[DELAY]: ' + str(delay) + '\t[SEED]: ' + str(seed))
            os.system("python run_dtrpo.py --env Pendulum --mode test --seed 0 --stochastic_delays " 
                      "--max_delay 50 --delay_proba " + str(delay) +
                      " --test_episodes 100 --test_steps 250 --save_dir " + save_dir)
