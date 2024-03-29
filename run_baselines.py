import fire 
import os


def run_trpo_delays(range_low=None, range_high=None,):
    import itertools
    from numpy import prod

    hyperparam = {
        # 'env' : ["Ant-v2", "HalfCheetah-v2", "Hopper-v2", "Humanoid-v2", "HumanoidStandup-v2", "Reacher-v2", "Swimmer-v2"],
        'env' : ["Hopper-v2", "Humanoid-v2", "HumanoidStandup-v2", "Reacher-v2", "Swimmer-v2"],
        # 'delay' : [0,5],
        'delay' : [5,],
    }

    n_runs = []
    for v in hyperparam.values():
        n_runs.append(len(v))
    print('Size of gird: {}'.format(prod(n_runs)))
    if range_low is None:
        range_low = 0 
    if range_high is None:
        range_high = prod(n_runs)
    print('Range ({}-{})'.format(range_low,range_high))

    names = list(hyperparam.keys())
    for i, values in enumerate(itertools.product(*hyperparam.values())):
        if i>=range_low and i<=range_high:
            print(list(zip(names,values)))
            os.system('python run_trpo.py --env "{0}" --mode train \
            --seeds 0 1 2 --delay {1} --epochs 1000 --steps_per_epoch 5000 \
            --max_ep_len 250 --delta 0.001 --v_hid 64 --v_l 1 --vf_lr 0.01 \
            --v_iters 3 --pi_hid 64 --pi_l 2 --save_period 100 \
            --save_dir "./output/{0}/trpo_delay_{1}"'.format(*values))


def run_m_trpo_delays(range_low=None, range_high=None,):
    import itertools
    from numpy import prod

    hyperparam = {
        'env' : ["Ant-v2", "HalfCheetah-v2", "Reacher-v2", "Swimmer-v2"],
        'delay' : [5,],
    }

    n_runs = []
    for v in hyperparam.values():
        n_runs.append(len(v))
    print('Size of gird: {}'.format(prod(n_runs)))
    if range_low is None:
        range_low = 0 
    if range_high is None:
        range_high = prod(n_runs)
    print('Range ({}-{})'.format(range_low,range_high))

    names = list(hyperparam.keys())
    for i, values in enumerate(itertools.product(*hyperparam.values())):
        if i>=range_low and i<=range_high:
            print(list(zip(names,values)))
            os.system('python run_trpo.py --env "{0}" --mode train \
            --seeds 0 1 2 3 4 5 4 6 7 8 9 --delay {1} --memoryless --epochs 2000 \
            --steps_per_epoch 5000 \
            --max_ep_len 250 --delta 0.001 --gamma 0.99 --v_hid 64 --v_l 1 --vf_lr 0.01 \
            --v_iters 3 --pi_hid 64 --pi_l 2 --save_period 100 \
            --save_dir "./output/{0}/m-trpo/trpo_delay_{1}"'.format(*values))




if __name__=='__main__':
    fire.Fire()