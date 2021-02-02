import os
import numpy as np

os.system("conda activate Thesis")
delays = [3, 5, 10, 15]
tests = [3, 5, 10, 15]

# DTRPO Deterministic Delay Pendulum Tests
for delay in delays:
    save_dir = "./output/l2trpo/Pendulum-Results-BIG/Results-Delay" + str(delay) + '-BIG'
    subfolders = [f.path for f in os.scandir(save_dir) if f.is_dir()]
    runs = len(subfolders)

    for test in tests:
        test_type = 'delay' + str(test)
        run = 0
        for folder in subfolders:
            seed = np.random.randint(10000)
            print('[METHOD]: DTRPO Delay' + str(delay) + '\t[DELAY]: ' + str(test) + '\t[SEED]: ' + str(seed) +
                  '\t[RUN]: ' + str(run) + '/' + str(runs))
            os.system('python run_dtrpo.py --env=Pendulum-v0 --mode=test --seeds ' + str(seed) +
                      ' --delay=' + str(test) +
                      ' --epoch_load=2000 --test_episodes=50 --test_steps=250 --test_type=' + test_type +
                      ' --save_dir ' + folder)
            run += 1
