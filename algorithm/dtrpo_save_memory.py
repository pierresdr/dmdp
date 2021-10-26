import warnings
import numpy as np
import torch.distributions
import utils.DTRPOCore as Core
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from torch.optim import Adam
from torch.nn import MSELoss, L1Loss
from datetime import datetime as dt
from datetime import timedelta
from utils.various import *

import gc


torch.backends.cudnn.enabled = False

EPS = 1e-8



        


def compute_loss_pi(ac, data, device):
    # Policy loss
    _, logp = ac.pi(data['obs'].to(device), data['act'].to(device))
    ratio = torch.exp(logp - data['logp'])
    loss_pi = -(ratio * data['adv']).mean()
    return loss_pi

def compute_loss_v(ac, data, device):
    return ((ac.v(data['obs'].to(device)) - data['ret'].to(device)) ** 2).mean()


def compute_loss_enc(ac, data, save_period, epoch, device, save_path):
    """Compute the loss of the belief stochastic for deterministic env.
    """
    u, log_probs = ac.enc.log_probs(
        data['extended_states'].to(device), 
        data['hidden_states'].to(device), 
        torch.from_numpy(data['mask']).to(device))
    if epoch % save_period == 0:
        save_noise(u, epoch, save_path)
        save_proba(log_probs, epoch, save_path)
    return -log_probs.mean()

def save_proba(log_probs, epoch, save_path):
    fig, ax = plt.subplots(1, 1, figsize=(6, 5))
    ax.hist(torch.exp(log_probs).detach().numpy())
    plt.savefig(os.path.join(save_path,str(epoch)+'_proba.png'))
    plt.close(fig)

def save_hidden_state(ac, obs, device, epoch, save_path):
    num_samples = min(obs.size(0), 100)
    with torch.no_grad():
        obs = ac.enc(obs.to(device)).detach()
    obs = obs[:num_samples]
    fig, ax = plt.subplots(1, 1, figsize=(6, 5))
    ax.hist(obs.detach().numpy())
    plt.savefig(os.path.join(save_path, str(epoch)+ '_hidden_state.png'))
    plt.close(fig)

def save_belief(ac, obs, device, epoch, save_path):
    num_samples = min(obs.size(0),100)
    with torch.no_grad():
            cond = ac.enc.get_cond(obs.to(device)).detach()
    cond = cond[:num_samples]
    samples = ac.enc.maf_proba.sample(num_samples=num_samples, cond_inputs=cond)
    fig, ax = plt.subplots(1, 1, figsize=(6, 5))
    ax.hist(samples[:, -1, :].detach().numpy())
    plt.savefig(os.path.join(save_path ,str(epoch)+'_belief.png'))
    plt.close(fig)

def save_noise(u, epoch, save_path):
    fig, ax = plt.subplots(1, 1, figsize=(6, 5))
    u = torch.cat((u, torch.normal(torch.zeros(u.size(0))).reshape(-1, 1)), 1)
    ax.hist(u.detach().numpy(), range=(-4, 4))
    plt.savefig(os.path.join(save_path, str(epoch)+ '_noise.png'))
    plt.close(fig)

def save_obs_density(obs, epoch, save_path):
    fig, ax = plt.subplots(1, 1, figsize=(6, 5))
    ax.hist(obs.detach().numpy())
    plt.savefig(os.path.join(save_path, str(epoch)+ '_encoded_state.png'))
    plt.close(fig)

def compute_kl(ac, data, old_pi, device):
    pi, _ = ac.pi(data['obs'].to(device), data['act'].to(device))
    kl_loss = torch.distributions.kl_divergence(pi, old_pi).mean()
    return kl_loss

@torch.no_grad()
def compute_kl_loss_pi(ac, data, old_pi, device):
    # Policy loss
    pi, logp = ac.pi(data['obs'].to(device), data['act'].to(device))
    ratio = torch.exp(logp - data['logp'])
    loss_pi = -(ratio * data['adv']).mean()
    kl_loss = torch.distributions.kl_divergence(pi, old_pi).mean()
    return loss_pi, kl_loss

def hessian_vector_product(ac, data, old_pi, v, damping_coeff, device):
    kl = compute_kl(ac, data, old_pi, device)

    grads = torch.autograd.grad(kl, ac.pi.parameters(), create_graph=True)
    flat_grad_kl = Core.flat_grads(grads)

    kl_v = (flat_grad_kl * v).sum()
    grads = torch.autograd.grad(kl_v, ac.pi.parameters())
    flat_grad_grad_kl = Core.flat_grads(grads)

    return flat_grad_grad_kl + v * damping_coeff

def update(ac, buf, enc_losses, v_losses, enc_optimizer, maf_optimizer,
            train_enc_iters, save_period, device, epoch, train_v_iters, vf_optimizer,
            delta, cg_iters, backtrack_iters, backtrack_coeff, damping_coeff,
            pretrain=False, stop_belief_training=False, save_path='test'):
    # If Pretraining then optimize only the Encoder
    if pretrain:
        # Encoder Update
        enc_losses.append(update_enc(ac, buf, enc_optimizer, maf_optimizer,
            train_enc_iters, save_period, epoch, device, save_path=save_path))
        # Value Function Loss Compatibility
        v_losses.append(np.nan)
        buf.reset()
    # Stop training belief module
    elif stop_belief_training:
        enc_losses.append(np.nan)
        data = buf.get()
        with torch.no_grad():
            data['obs'] = ac.enc(data['obs'].to(device)).detach()
        # Policy Update
        update_pi(ac, data, device, delta, cg_iters, backtrack_iters, backtrack_coeff, damping_coeff)
        # Value Function Update
        v_loss = update_v(ac, data, train_v_iters, vf_optimizer, device)
        v_losses.append(v_loss)
    # Else optimize Policy and Value Function
    else:
        enc_losses.append(update_enc(ac, buf, enc_optimizer, maf_optimizer,
            train_enc_iters, save_period, epoch, device, save_path=save_path))
        # Extract Encoder Prediction once for all the Networks that needs it
        data = buf.get()
        with torch.no_grad():
            data['obs'] = ac.enc(data['obs'].to(device)).detach()
        # Policy Update
        update_pi(ac, data, device, delta, cg_iters, backtrack_iters, backtrack_coeff, damping_coeff)
        # Value Function Update
        v_loss = update_v(ac, data, train_v_iters, vf_optimizer, device)
        v_losses.append(v_loss)
    return enc_losses, v_losses

def update_pi(ac, data, device, delta, cg_iters, backtrack_iters, backtrack_coeff, damping_coeff):
    # Compute old pi distribution
    with torch.no_grad():
        old_pi, _ = ac.pi(data['obs'].to(device), data['act'].to(device))

    pi_loss = compute_loss_pi(ac, data, device)
    pi_l_old = pi_loss.item()

    grads = Core.flat_grads(torch.autograd.grad(pi_loss, ac.pi.parameters()))

    # Core calculations for TRPO
    Hx = lambda v: hessian_vector_product(ac, data, old_pi, v, damping_coeff, device)
    x = Core.conjugate_gradients(Hx, grads, cg_iters)

    alpha = torch.sqrt(2 * delta / (torch.matmul(x, Hx(x)) + EPS))

    old_params = Core.get_flat_params_from(ac.pi)

    def set_and_eval(step):
        new_params = old_params - alpha * x * step
        Core.set_flat_params_to(ac.pi, new_params)
        loss_pi, kl_loss = compute_kl_loss_pi(ac, data, old_pi, device)
        return kl_loss.item(), loss_pi.item()

    # TRPO augments npg with backtracking line search, hard kl
    for j in range(backtrack_iters):
        kl, pi_l_new = set_and_eval(step=backtrack_coeff ** j)

        if kl <= delta and pi_l_new <= pi_l_old:
            prGreen('\tAccepting new params at step %d of line search.' % j)
            break

        if j == backtrack_iters - 1:
            prRed('\tLine search failed! Keeping old params.')
            kl, pi_l_new = set_and_eval(step=0)


def update_enc(ac, buf, enc_optimizer, maf_optimizer,
            train_enc_iters, save_period, epoch, device, save_path):
    # Encoder Learning
    loss_enc = None
    ac.enc.train()
    for i in range(train_enc_iters):
        enc_optimizer.zero_grad()
        maf_optimizer.zero_grad()
        data = buf.get_pred_data()
        loss_enc = compute_loss_enc(ac, data, save_period, epoch, device, save_path=save_path)
        loss_enc.backward(retain_graph=True)
        enc_optimizer.step()
        maf_optimizer.step()

    ac.enc.eval()
    loss_enc_item  = loss_enc.detach().item()
    return loss_enc_item

def update_v(ac, data, train_v_iters, vf_optimizer, device):
    # Value Function Learning
    for i in range(train_v_iters):
        vf_optimizer.zero_grad()
        loss_v = compute_loss_v(ac, data, device)
        loss_v.backward()
        vf_optimizer.step()
    loss_v_item = loss_v.detach().item()
    return loss_v_item

def format_o(o, env, act_dim=None):
    if isinstance(env.action_space, Discrete):
        assert act_dim is not None, "Must provide action dimension to format_o"
        # Create one hot encoding of the actions contained in the state
        temp_o = torch.tensor([i % act_dim == o[1][i // act_dim]
                                for i in range(act_dim * len(o[1]))]).float()
        return torch.cat((torch.tensor(o[0]), temp_o.reshape(-1)))
    else:
        return np.hstack((o[0], o[1].reshape(-1)))



def step_training(env, ac, buf, o, t, episode, ep_rewards, ep_lengths, ep_ret, ep_len, 
    pretrain, device, max_ep_len, steps_per_epoch):
    # Select a new action
    a, v, logp = ac.step(torch.as_tensor(o, dtype=torch.float32).unsqueeze(dim=0).to(device))

    # Execute the action
    next_o, r, d, info = env.step(a.reshape(-1))
    next_o = format_o(next_o, env)
    ep_ret += np.sum(r)
    ep_len += 1

    # Save the visited transition in to the Buffer
    buf.store(o, a, np.sum(r), v, not d, info, logp, episode, pretrain=pretrain)
    o = next_o

    # Is the episode terminated and why?
    timeout = ep_len == max_ep_len
    terminal = d or timeout
    epoch_ended = t == steps_per_epoch - 1

    if terminal or epoch_ended:
        # If Epoch ended before Episode could end
        if epoch_ended and not terminal:
            prGreen('\tWarning: trajectory cut off by epoch at %d steps.' % ep_len)
        # If Episode didn't reach terminal state, bootstrap value target of the reached state
        if timeout or epoch_ended:
            _, v, _ = ac.step(torch.as_tensor(o, dtype=torch.float32).unsqueeze(dim=0).to(device))
        # If the Trajectory ended by its own, set State Value to 0
        else:
            v = 0
        # Let the Buffer adjust itself when an Episode has ended
        buf.finish_path(v)
        if terminal:
            # Only print EpRet and EpLen if Episode finished (otherwise they're not indicative)
            ep_rewards.append(ep_ret)
            ep_lengths.append(ep_len)

        # Prepare for interaction with environment for the next episode:
        # Start recording timings, reset the environment to get s_0
        o, ep_ret, ep_len = env.reset(), 0, 0
        o = format_o(o, env)

        # Update Episode counting variable
        episode += 1
    return buf, o, episode, episode, ep_rewards, ep_lengths, ep_ret, ep_len


def train_inside_loop(env, ac, buf, o, epoch, epochs_belief_training, pretrain_epochs, 
        start_time, ep_ret, ep_len, avg_reward, std_reward, avg_length, timings, device,
        enc_optimizer, maf_optimizer, v_losses, enc_losses, train_enc_iters,
        delta, cg_iters, backtrack_iters, backtrack_coeff, damping_coeff,
        vf_optimizer, train_v_iters,
        max_ep_len, steps_per_epoch, pretrain_steps, save_period, save_path='test'):
    # If Pre-Training of Belief Module is (still) active:
    if epoch < pretrain_epochs:
        pretrain = True
        max_epoch_steps = pretrain_steps
    else:
        pretrain = False
        max_epoch_steps = steps_per_epoch

    # If Belief Module completed its training:
    if epoch > epochs_belief_training:
        stop_belief_training = True
    else:
        stop_belief_training = False

    # Reset Episodes variables
    ep_rewards = []
    ep_lengths = []
    episode = 0

    for t in range(max_epoch_steps):
        buf, o, episode, episode, ep_rewards, ep_lengths, ep_ret, ep_len = step_training(
                env=env, ac=ac, buf=buf, o=o, t=t, episode=episode, ep_rewards=ep_rewards, 
                ep_lengths=ep_lengths, ep_ret=ep_ret, ep_len=ep_len, pretrain=pretrain,
                device=device, max_ep_len=max_ep_len, steps_per_epoch=steps_per_epoch)
        

    # Perform TRPO update at the end of the Epoch
    enc_losses, v_losses= update(ac=ac, buf=buf, enc_losses=enc_losses, v_losses=v_losses, 
            enc_optimizer=enc_optimizer, maf_optimizer=maf_optimizer, vf_optimizer=vf_optimizer,
            train_enc_iters=train_enc_iters, save_period=save_period, train_v_iters=train_v_iters, 
            delta=delta, cg_iters=cg_iters, backtrack_iters=backtrack_iters, backtrack_coeff=backtrack_coeff,
            device=device, epoch=epoch, pretrain=pretrain, damping_coeff=damping_coeff,
            stop_belief_training=stop_belief_training, save_path=save_path)

    # Record Timings
    elapsed_time = dt.now() - start_time

    # Gather Epoch results and print them
    avg_reward.append(np.mean(ep_rewards))
    std_reward.append(np.std(ep_rewards))
    avg_length.append(np.mean(ep_lengths))
    timings.append(elapsed_time)
    print_update(avg_reward, enc_losses, v_losses, elapsed_time, epoch)

    # Save Epoch Results each "save_period" epochs
    if epoch % save_period == 0:
        save_session(ac, avg_reward=avg_reward, std_reward=std_reward, enc_losses=enc_losses, 
                v_losses=v_losses, timings=timings, elapsed_time=elapsed_time, epoch=epoch, 
                save_path=save_path)

    # Plot all the data of this Epoch
    save_results(avg_reward, std_reward, enc_losses, v_losses, epoch, save_path=save_path)
    return buf, o, ep_ret, ep_len, avg_reward, std_reward, avg_length, timings 
    

def train(env, ac, epochs, pretrain_epochs, epochs_belief_training, train_continue,
        enc_optimizer, maf_optimizer, v_losses, vf_optimizer, train_v_iters,
        enc_losses, train_enc_iters, buf, damping_coeff,
        delta, cg_iters, backtrack_iters, backtrack_coeff,
        max_ep_len, steps_per_epoch, pretrain_steps, timings,
        save_period, save_path, avg_reward, std_reward, avg_length,
        device,):
    # Load previous training final data in order to continue from there
    if train_continue:
        load_session()

    # Prepare for interaction with environment for the first Episode:
    # Start recording timings, reset the environment to get s_0
    start_time = dt.now()
    o, ep_ret, ep_len = env.reset(), 0, 0
    o = format_o(o, env)

    # ---- TRAINING LOOP ----
    for epoch in range(1, epochs + 1):
        buf, o, ep_ret, ep_len, avg_reward, std_reward, avg_length, timings = train_inside_loop(
                env=env, ac=ac, o=o, epoch=epoch, epochs_belief_training=epochs_belief_training,
                pretrain_epochs=pretrain_epochs, delta=delta, cg_iters=cg_iters, 
                backtrack_iters=backtrack_iters, backtrack_coeff=backtrack_coeff,
                start_time=start_time, ep_ret=ep_ret, ep_len=ep_len, avg_reward=avg_reward, 
                std_reward=std_reward, avg_length=avg_length, timings=timings, device=device,
                enc_optimizer=enc_optimizer, maf_optimizer=maf_optimizer, v_losses=v_losses,
                enc_losses=enc_losses, train_enc_iters=train_enc_iters, buf=buf, 
                damping_coeff=damping_coeff, vf_optimizer=vf_optimizer, train_v_iters=train_v_iters,
                max_ep_len=max_ep_len, steps_per_epoch=steps_per_epoch, pretrain_steps=pretrain_steps,
                save_period=save_period, save_path=save_path,)

        gc.collect()

        

def print_update(avg_reward, enc_losses, v_losses, elapsed_time, epoch,):
    update_message = '[EPOCH]: {0}\t[AVG. REWARD]: {1:.4f}\t[ENC. LOSS]: {2:.4f}\t[V LOSS]: {3:.4f}\t[ELAPSED TIME]: {4}'
    elapsed_time_str = ''.join(str(elapsed_time).split('.')[0])
    format_args = (epoch, avg_reward[-1], enc_losses[-1], v_losses[-1], elapsed_time_str)
    prYellow(update_message.format(*format_args))

def save_session(ac, avg_reward, std_reward, enc_losses, v_losses,
            timings, elapsed_time, epoch, save_path='test'):
    if not os.path.exists(save_path):
        os.mkdir(save_path)

    save_path = os.path.join(save_path, 'model_'+str(epoch)+'.pt')

    ckpt = {'policy_state_dict': ac.pi.state_dict(),
            'value_state_dict': ac.v.state_dict(),
            'encoder_state_dict': ac.enc.state_dict(),
            'avg_reward': avg_reward,
            'std_reward': std_reward,
            'enc_losses': enc_losses,
            'v_losses': v_losses,
            'epoch': epoch,
            'timings': timings,
            'elapsed_time': elapsed_time
            }

    torch.save(ckpt, save_path)

def load_session(ac, epoch=None, save_path='test'):
    if epoch is None:
        files_name = os.listdir(save_path)
        files_name.remove('model_parameters.txt')
        models = list(filter(lambda n: "model" in n,  files_name))
        if len(models)==1:
            load_path = os.path.join(save_path, 'model.pt')
        else: 
            temp = np.array([int(name.replace('model_','').replace('.pt','')) for name in iter(models)])
            load_path = os.path.join(save_path, 'model_'+str(max(temp)-1)+'.pt')
    else:
        load_path = os.path.join(save_path, 'model_'+str(epoch)+'.pt')
    
    ckpt = torch.load(load_path)

    ac.pi.load_state_dict(ckpt['policy_state_dict'])
    ac.v.load_state_dict(ckpt['value_state_dict'])
    ac.enc.load_state_dict(ckpt['encoder_state_dict'])
    avg_reward = ckpt['avg_reward']
    std_reward = ckpt['std_reward']
    enc_losses = ckpt['enc_losses']
    v_losses = ckpt['v_losses']
    epoch = ckpt['epoch']
    timings = ckpt['timings']
    elapsed_time = ckpt['elapsed_time']
    return avg_reward, std_reward, enc_losses, v_losses, epoch, timings, elapsed_time

def save_results(avg_reward, std_reward, enc_losses, v_losses, epoch, save_path='test'):
    x = range(0, epoch, 1)
    errorbar_plot(x, avg_reward, xlabel='Epochs', ylabel='Mean Reward', filename='reward',
                        error=std_reward,save_path=save_path)
    scatter_plot(x, avg_reward, xlabel='Epochs', ylabel='Mean Reward', filename='reward_scatter',
            save_path=save_path)
    scatter_plot(x, enc_losses, xlabel='Epochs', ylabel='Encoder Loss', filename='enc_loss_scatter',
            save_path=save_path)
    scatter_plot(x, v_losses, xlabel='Epochs', ylabel='Value Loss', filename='v_loss_scatter',
            save_path=save_path)

def scatter_plot(x, y, xlabel='Epochs', ylabel='Mean Reward', filename='reward', save_path='test'):
    fig, ax = plt.subplots(1, 1, figsize=(6, 5))
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    ax.scatter(x, y)
    plt.savefig(os.path.join(save_path, filename + '.png'))
    plt.close(fig)

def errorbar_plot(x, y, xlabel='Epochs', ylabel='Mean Reward', filename='reward', error=None, 
            save_path='test'):
    fig, ax = plt.subplots(1, 1, figsize=(6, 5))
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    ax.errorbar(x, y, yerr=error, fmt='-o')
    plt.savefig(os.path.join(save_path, filename + '.png'))
    plt.close(fig)

