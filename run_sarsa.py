from importlib import import_module
from utils.various import get_output_folder
from algorithm.sarsa import SARSA
import os
import json

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Trust Region Policy Optimization (PyTorch)')

    # General Arguments for Training and Testing TRPO
    parser.add_argument('--mode', default='train', type=str, choices=['train', 'test'])
    parser.add_argument('--env', default='MountainCarDelayEnv', type=str, choices=['CartPoleDelayEnv',
                                                                                   'MountainCarDelayEnv',
                                                                                   'PendulumDelayEnv'])
    parser.add_argument('--delay', type=int, default=30, help='Number of Delay Steps for the Environment.')
    parser.add_argument('--seed', type=int, default=0, help='Seed for Reproducibility purposes.')
    parser.add_argument('--train_render', action='store_true', help='Whether render the Env during training or not.')
    parser.add_argument('--train_render_ep', type=int, default=1, help='Which episodes render the env during training.')

    # Train Specific Arguments
    parser.add_argument('--steps_per_epoch', type=int, default=12500, help='Number of Steps per Epoch.')
    parser.add_argument('--epochs', type=int, default=200, help='Number of Epochs of Training.')
    parser.add_argument('--max_ep_len', type=int, default=2500, help='Max Number of Steps per Episode')

    # Test Specific Arguments
    parser.add_argument('--test_episodes', type=int, default=10, help='Number of Test Episodes.')
    parser.add_argument('--test_steps', type=int, default=250, help='Number of Steps per Test Episode.')

    # SARSA Specific Arguments
    parser.add_argument('--gamma', type=float, default=0.99, help='Discount Factor.')
    parser.add_argument('--lam', type=float, default=0.9, help='Eligibility Traces Factor.')
    parser.add_argument('--lr', type=float, default=0.3, help='Learning Rate.')
    parser.add_argument('--e', type=float, default=0.2, help='E-Greedy Policy Parameter.')

    # Discretization Specific Arguments
    parser.add_argument('--s_space', type=int, default=10, help='State Space Discretization Grid Dimension.')
    parser.add_argument('--a_space', type=int, default=3, help='Action Space Discretization.')

    # Folder Management Arguments
    parser.add_argument('--save_dir', default='./output/sarsa', type=str, help='Output folder for the Trained Model')
    args = parser.parse_args()

    # Environment Initialization
    env = import_module('env.' + args.env)
    env = getattr(env, args.env)

    # ---- TRAIN MODE ---- #
    if args.mode == 'train':
        args.save_dir = get_output_folder(os.path.join(args.save_dir, args.env+'-Results'), args.env)
        with open(os.path.join(args.save_dir, 'model_parameters.txt'), 'w') as text_file:
            json.dump(args.__dict__, text_file, indent=2)

        agent = SARSA(env, delay=args.delay, seed=args.seed, epochs=args.epochs, steps=args.steps_per_epoch,
                      max_steps=args.max_ep_len, lam=args.lam, gamma=args.gamma, lr=args.lr, e=args.e,
                      s_space=args.s_space, a_space=args.a_space, save_dir=args.save_dir,
                      train_render=args.train_render, train_render_ep=args.train_render_ep)

        agent.train()

    # ---- TEST MODE ---- #
    elif args.mode == 'test':
        args.save_model = next(filter(lambda x: '.pt' in x, os.listdir(args.save_dir)))
        model_path = os.path.join(args.save_dir, args.save_model)
        load_parameters = os.path.join(args.save_dir, 'model_parameters.txt')
        with open(load_parameters) as text_file:
            file_args = json.load(text_file)

        agent = SARSA(env, delay=args.delay, save_dir=args.save_dir)

        agent.test(test_episodes=args.test_episodes, max_steps=args.test_steps)