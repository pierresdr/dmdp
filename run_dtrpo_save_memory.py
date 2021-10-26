import json, argparse
import gym#, gym_puddle
import torch.nn as nn
from importlib import import_module
from utils import DTRPOCore as Core
from utils.various import *
from utils.delays import DelayWrapper
from utils.stochastic_wrapper import StochActionWrapper
import gym_bicycle
from torch.optim import Adam
from utils.DTRPOBuffer import GAEBufferDeter
from algorithm.dtrpo_save_memory import train


torch.backends.cudnn.enabled = False

def launch_dtrpo(args, seed):
    # ---- ENV INITIALIZATION ----
    env = gym.make(args.env)
    env.seed(seed)

    # Add stochasticity wrapper
    if args.force_stoch_env:
        env = StochActionWrapper(env, distrib=args.stoch_mdp_distrib, param=args.stoch_mdp_param)
    update_message = 'Created env with observation space {} and action space {}'.format(
            env.observation_space,  env.action_space)
    prYellow(update_message)

    # Add the delay wrapper
    env = DelayWrapper(env, delay=args.delay, max_delay=args.max_delay)
    update_message = 'Running env {} with initial delay {}'.format(args.env, args.delay,)
    prYellow(update_message)


    # ---- TRAIN MODE ----
    if args.mode == 'train':
        # Create output folder and save training parameters
        save_path = get_output_folder(args.save_dir, args.env)
        with open(os.path.join(save_path, 'model_parameters.txt'), 'w') as text_file:
            json.dump(args.__dict__, text_file, indent=2)

        torch.manual_seed(seed)
        np.random.seed(seed)

        obs_dim = get_space_dim(env.observation_space)
        act_dim = get_space_dim(env.action_space)
        state_dim = get_space_dim(env.state_space)

        # Set device
        if args.device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            device = torch.device(args.device)
        prLightPurple('Device: {}'.format(device))

        ac_kwargs = dict(
            pi_hidden_sizes=[args.pi_hid] * args.pi_l,
            v_hidden_sizes=[args.v_hid] * args.v_l,
            enc_dim=args.enc_dim, enc_heads=args.enc_heads, enc_ff=args.enc_ff, enc_l=args.enc_l,
            enc_rescaling=args.enc_rescaling, enc_causal=args.enc_causal, pred_to_pi=args.enc_pred_to_pi,
            hidden_dim=args.hidden_dim, n_blocks_maf=args.n_blocks_maf, hidden_dim_maf=args.hidden_dim_maf,
            lstm=args.lstm, n_layers=args.n_layers, hidden_size=args.hidden_size, conv=args.convolutions,
            only_last_belief=args.only_last_belief,
            activation=eval(args.pi_activation)
        )
        ac = Core.TRNActorCritic(obs_dim, env.action_space, env.state_space, use_belief=True, **ac_kwargs).to(device)
        var_counts = tuple(Core.count_vars(module) for module in [ac.pi, ac.v, ac.enc])
        prLightPurple('Number of parameters: \t pi: %d, \t v: %d, \t enc: %d\n' % var_counts)

        enc_optimizer = Adam(ac.enc.encoder.parameters(), lr=args.enc_lr)
        maf_optimizer = Adam(ac.enc.maf_proba.parameters(), lr=args.maf_lr)
        vf_optimizer = Adam(ac.v.parameters(), lr=args.vf_lr)

        buf = GAEBufferDeter(obs_dim, state_dim, env.action_space.shape, act_dim, args.size_pred_buf, args.batch_size_pred,
                args.steps_per_epoch, args.gamma, args.lam)

        avg_reward = []
        std_reward = []
        avg_length = []
        enc_losses = []
        v_losses = []
        timings = []
        train(env, ac, epochs=args.epochs, pretrain_epochs=args.pretrain_epochs, 
                epochs_belief_training=args.epochs_belief_training, train_continue=False,
                enc_optimizer=enc_optimizer, maf_optimizer=maf_optimizer, v_losses=v_losses,
                enc_losses=enc_losses, train_enc_iters=args.train_enc_iters, buf=buf,
                max_ep_len=args.max_ep_len, steps_per_epoch=args.steps_per_epoch, 
                pretrain_steps=args.pretrain_steps, avg_reward=avg_reward,
                 vf_optimizer=vf_optimizer, train_v_iters=args.v_iters,
                std_reward=std_reward, avg_length=avg_length, timings=timings,
                delta=args.delta, cg_iters=args.cg_iters, backtrack_iters=args.backtrack_iters, 
                backtrack_coeff=args.backtrack_coeff, damping_coeff=args.damping_coeff,
                save_period=args.save_period, save_path=save_path, device=device)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Trust Region Policy Optimization (PyTorch)')

    # General Arguments for Training and Testing TRPO
    parser.add_argument('--mode', default='train', type=str, choices=['train', 'test'])
    parser.add_argument('--env', default='Pendulum-v0', type=str)
    parser.add_argument('--device', type=str, default=None, help='Device on which to run code.')

    parser.add_argument('--seeds', nargs='+', type=int, default=0, help='Seed for Reproducibility purposes.')
    parser.add_argument('--curr_seed', type=int, default=0, help='Seed of the current run for parameter saving.')
    parser.add_argument('--delay', type=int, default=3, help='Number of Delay Steps for the Environment.')
    parser.add_argument('--max_delay', default=50, type=int, help='Maximum delay of the environment.')
    parser.add_argument('--delay_proba', type=float, default=0.7, help='Probability of Observation for the Delay Process.')
    parser.add_argument('--force_stoch_env', action='store_true', help='Force the env to be stochastic.')
    parser.add_argument('--stoch_mdp_param', type=float, default=1, help='Depending on the stochasticity of the action:'
                                                                         + '- Gaussian: STD of the Distribution\n'
                                                                         + '- Uniform: Probability of sampling from\n'
                                                                         + '- LogNormal: STD of the Distribution\n'
                                                                         + '- Triangular: Mode of the "triangle"\n'
                                                                         + '- Quadratic: A and B parameters of Beta\n'
                                                                         + '- U-Shaped: A and B parameters of Beta\n'
                                                                         + '- Beta: A=8, B=2. No parameters.')
    parser.add_argument('--stoch_mdp_distrib', default='Gaussian', type=str, choices=['Gaussian', 'Uniform',
                                                                                      'LogNormal', 'Triangular',
                                                                                      'Quadratic', 'U-Shaped', 'Beta'],
                        help='Type of distribution of the action noise.')

    # Train Specific Arguments
    parser.add_argument('--steps_per_epoch', type=int, default=5000, help='Number of Steps per Epoch.')
    parser.add_argument('--epochs', type=int, default=200, help='Number of Epochs of Training.')
    parser.add_argument('--max_ep_len', type=int, default=250, help='Max Number of Steps per Episode')
    parser.add_argument('--gamma', type=float, default=0.99, help='Discount Factor.')
    parser.add_argument('--delta', type=float, default=0.1, help='TRPO Max KL Divergence.')
    parser.add_argument('--save_period', type=int, default=1, help='Save models learned parameters every save_period.')
    parser.add_argument('--train_continue', action='store_true', help='Continue the training from save_dir folder.')

    # Test Specific Arguments
    parser.add_argument('--test_episodes', type=int, default=10, help='Number of Test Episodes.')
    parser.add_argument('--test_steps', type=int, default=250, help='Number of Steps per Test Episode.')
    parser.add_argument('--epoch_load', type=str, default=None, help='Epoch to load.')
    parser.add_argument('--test_type', type=str, default=None, help='Test Name for saving pt file.')

    # Value Function Specific Arguments
    parser.add_argument('--v_hid', type=int, default=64, help='Number of Neurons in each Hidden Layers.')
    parser.add_argument('--v_l', type=int, default=1, help='Number of Hidden Layers in each Network.')
    parser.add_argument('--vf_lr', type=float, default=0.01, help='Value Function Adam Learning Rate.')
    parser.add_argument('--v_iters', type=int, default=3, help='Value Function number of Iterations per Epoch.')

    # Policy Network Specific Arguments
    parser.add_argument('--pi_activation', default='nn.ReLU', type=str)
    parser.add_argument('--pi_hid', type=int, default=64, help='Number of Neurons in each Hidden Layers.')
    parser.add_argument('--pi_l', type=int, default=2, help='Number of Hidden Layers in each Network.')
    parser.add_argument('--damping_coeff', type=float, default=0.1, help='Numerical stability for Hessian Product')
    parser.add_argument('--cg_iters', type=int, default=10, help='CG Iterations for Hessian Product')
    parser.add_argument('--backtrack_iters', type=int, default=10, help='Max Backtracking Iterations for Line Search')
    parser.add_argument('--backtrack_coeff', type=float, default=0.8, help='Distance for each Backtracking Iteration')

    # Generalized Advantage Estimation Specific Arguments
    parser.add_argument('--lam', type=float, default=0.97, help='Lambda Coefficient for GAE.')

    # Train Transformer Encoder Specific Arguments
    parser.add_argument('--pretrain_epochs', type=int, default=50, help='Epochs for Encoder Pre-Training.')
    parser.add_argument('--epochs_belief_training', type=int, default=200, help='Epochs for Encoder Pre-Training.')
    parser.add_argument('--pretrain_steps', type=int, default=5000, help='Epochs for Encoder Pre-Training.')
    parser.add_argument('--train_enc_iters', type=int, default=1, help='Encoder Adam Optimizer iterations per epoch.')
    parser.add_argument('--size_pred_buf', type=int, default=100000, help='Size of the prediction buffer.')
    parser.add_argument('--batch_size_pred', type=int, default=5000, help='Batch size for the prediction training.')

    # Transformer Encoder Specific Arguments
    parser.add_argument('--enc_lr', type=float, default=5e-3, help='Encoder Adam Optimizer Learning Rate.')
    parser.add_argument('--maf_lr', type=float, default=5e-3, help='Encoder MAF Adam Optimizer Learning Rate.')
    parser.add_argument('--enc_dim', type=int, default=64, help='Encoder Dimension.')
    parser.add_argument('--enc_heads', type=int, default=2, help='Encoder heads for Multi-Attention.')
    parser.add_argument('--enc_l', type=int, default=1, help='Encoder number of layers.')
    parser.add_argument('--enc_ff', type=int, default=8, help='Encoder FeedForward layer dimension.')
    parser.add_argument('--enc_rescaling', action='store_true', help='Whether activate State Rescaling or not.')
    parser.add_argument('--enc_causal', action='store_true', help='Whether using a Causal Enc. or Standard Enc.')
    parser.add_argument('--enc_pred_to_pi', action='store_true', help='Whether feeding Pi with Prediction or Encoded State.')
    parser.add_argument('--only_last_belief', action='store_true', help='Learn only the last belief distribution.')

    # Convolution pre-processing
    parser.add_argument('--convolutions', action='store_true', help='Whether to pre-process input with a convolution.')

    # LSTM Encoder Specific Arguments (Only Deterministic Delays)
    parser.add_argument('--lstm', action='store_true', help='Use LSTM encoder.')
    parser.add_argument('--n_layers', type=int, default=3, help='.')
    parser.add_argument('--hidden_size', type=int, default=16, help='.')

    # Masked Autoregressive Flow Specific Arguments
    parser.add_argument('--n_blocks_maf', default=5, type=int, help='Number of MAF Layers')
    parser.add_argument('--hidden_dim', default=8, type=int, help='Number of Encoder Layers')
    parser.add_argument('--hidden_dim_maf', default=16, type=int, help='Number of Encoder Layers')

    # Folder Management Arguments
    parser.add_argument('--save_dir', default='./output/dtrpo', type=str, help='Output folder for the Trained Model')
    args = parser.parse_args()

    print(args.save_dir)
    for i in list(args.seeds):
        print('Launching Seed: ' + str(i))
        args.curr_seed = i
        launch_dtrpo(args, i)