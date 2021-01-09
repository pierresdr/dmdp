import os
import numpy as np

os.system("conda activate Thesis")
algos = ['Augmented', 'Memoryless']
delays = [3, 5, 10, 15, 20]

for algo in algos:
    for delay in delays:
        save_dir = './output/trpo/Pendulum-' + algo + '/Results-Delay' + str(delay)
        subfolders = [f.path for f in os.scandir(save_dir) if f.is_dir()]
        runs = len(subfolders)
        run = 0

        for folder in subfolders:
            print(folder)
            seed = np.random.randint(10000)
            print('[METHOD]: TRPO\t[DELAY]: ' + str(delay) + '\t[SEED]: ' + str(seed) + '\t[RUN]: ' + str(run) + '/' + str(runs))
            os.system('python run_trpo.py --env Pendulum-v0 --mode test --seeds ' + str(seed) + ' --delay ' + str(delay) +
                      ' --test_epoch 2000 --test_episodes 50 --test_steps 250 --save_dir ' + folder)
            run += 1