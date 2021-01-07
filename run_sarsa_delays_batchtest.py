import os

os.system("conda activate Thesis")

seeds = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
delays = [0, 3, 5, 10, 15, 20]


for delay in delays:
    for seed in seeds:
        save_dir = "./output/sarsa/Pendulum-AISTATS/Results-Delay" + str(delay) \
                   + "/PendulumDelayEnv-run" + str(seed)

        print('[METHOD]: SARSA\t[DELAY]: ' + str(delay) + '\t[SEED]: ' + str(seed))
        os.system("python run_sarsa.py --env Pendulum-AISTATS --mode test --seed 0 --delay " + str(delay) +
                  " --test_episodes 50 --test_steps 250 --save_dir " + save_dir)
