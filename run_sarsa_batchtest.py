import os
import numpy as np

os.system("conda activate Thesis")
delays = [3, 5, 10, 15, 20]

for delay in delays:
    save_dir = './output/dsarsa/Pendulum-Results/Results-Delay' + str(delay)
    subfolders = [f.path for f in os.scandir(save_dir) if f.is_dir()]
    runs = len(subfolders)
    run = 0

    for folder in subfolders:
        print(folder)
        seed = np.random.randint(10000)
        print('[METHOD]: DSARSA\t[DELAY]: ' + str(delay) + '\t[SEED]: ' + str(seed) + '\t[RUN]: ' + str(run) + '/' + str(runs))
        os.system('python run_sarsa.py --dsarsa --env Pendulum --mode test --seeds ' + str(seed) + ' --delay ' + str(delay) +
                  ' --test_episodes 50 --test_steps 250 --save_dir ' + folder)
        run += 1
