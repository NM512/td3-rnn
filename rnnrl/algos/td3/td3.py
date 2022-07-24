from ast import arg
from copy import deepcopy
import itertools
from re import A
import numpy as np
import torch
from torch.optim import Adam
from torch.utils.tensorboard import SummaryWriter
import gym
import time
from rnnrl.algos.td3 import core
from rnnrl.utils.logx import EpochLogger
from rnnrl.utils.wrappers import PartialObservation, StackedObservation


class ReplayBuffer:
    """
    A simple FIFO experience replay buffer for TD3 agents.
    """

    def __init__(self, obs_dim, act_dim, size, recurrent, hidden_dim, n_sequence, n_burn_in, net_names, device):
        self.obs_buf = np.zeros(core.combined_shape(size, obs_dim), dtype=np.float32)
        self.obs2_buf = np.zeros(core.combined_shape(size, obs_dim), dtype=np.float32)
        self.act_buf = np.zeros(core.combined_shape(size, act_dim), dtype=np.float32)
        self.rew_buf = np.zeros(size, dtype=np.float32)
        self.done_buf = np.zeros(size, dtype=np.float32)
        if recurrent:
            self.hidden_buf = {
                net:dict(
                    h = np.zeros(core.combined_shape(size, hidden_dim), dtype=np.float32),
                    c = np.zeros(core.combined_shape(size, hidden_dim), dtype=np.float32),
                    h2 = np.zeros(core.combined_shape(size, hidden_dim), dtype=np.float32),
                    c2 = np.zeros(core.combined_shape(size, hidden_dim), dtype=np.float32)
                )for net in net_names
            }
            
        self.ptr, self.size, self.max_size = 0, 0, size
        self.recurrent = recurrent
        self.n_sequence = n_sequence
        self.n_burn_in = n_burn_in
        self.net_names = net_names
        self.device = device

    def store(self, obs, act, rew, next_obs, done, hiddens, next_hiddens):
        self.obs_buf[self.ptr] = obs
        self.obs2_buf[self.ptr] = next_obs
        self.act_buf[self.ptr] = act
        self.rew_buf[self.ptr] = rew
        self.done_buf[self.ptr] = done
        if self.recurrent:
            for net in self.net_names:
                h, c = hiddens[net]
                h2, c2 = next_hiddens[net]
                # Detach the hidden state so that BPTT only goes through until this time step
                self.hidden_buf[net]['h'][self.ptr] = h.cpu().numpy()
                self.hidden_buf[net]['c'][self.ptr] = c.cpu().numpy()
                self.hidden_buf[net]['h2'][self.ptr] = h2.cpu().numpy()
                self.hidden_buf[net]['c2'][self.ptr] = c2.cpu().numpy()
            
        self.ptr = (self.ptr+1) % self.max_size
        self.size = min(self.size+1, self.max_size)

    def sample_batch(self, batch_size=32):
        if self.recurrent:
            burn_in_idxs = []
            idxs = []
            for _ in range(batch_size):
                done_included = True
                while done_included:
                    # -1 is for incorporating act2
                    idx = np.random.randint(0, self.size - self.n_burn_in - self.n_sequence -1)
                    # Avoid Done is in 0~n-1 of total sequence length
                    done_included = self.done_buf[
                        idx:idx + self.n_burn_in + self.n_sequence - 1].sum() > 0.0
                burn_in_idxs.append(np.arange(idx, idx+self.n_burn_in)[None,:])
                idxs.append(np.arange(
                    idx+self.n_burn_in, idx+self.n_burn_in+self.n_sequence)[None,:])
            
            # Shape (batch_size, sequence_length)
            burn_in_idxs = np.concatenate(burn_in_idxs, 0)
            idxs = np.concatenate(idxs, 0)

            batch = dict(
                # Shape (batch_size, step_length, vector_dim)
                obs=self.obs_buf[idxs],
                obs2=self.obs2_buf[idxs],
                act=self.act_buf[idxs],
                rew=self.rew_buf[idxs],
                done=self.done_buf[idxs],
                )
            if self.n_burn_in > 0:
                batch.update(
                    b_obs=self.obs_buf[burn_in_idxs],
                    b_obs2=self.obs2_buf[burn_in_idxs],
                    b_act=self.act_buf[burn_in_idxs],
                    b_act2=self.act_buf[burn_in_idxs+1],
                    )
                for net in self.net_names:
                    batch.update({
                        # give hidden state for burn-in
                        # Shape (1, batch_size, hidden_dim)
                        net+'_b_h':self.hidden_buf[net]['h'][burn_in_idxs[:,0]][None,:],
                        net+'_b_c':self.hidden_buf[net]['c'][burn_in_idxs[:,0]][None,:],
                        net+'_b_h2':self.hidden_buf[net]['h2'][burn_in_idxs[:,0]][None,:],
                        net+'_b_c2':self.hidden_buf[net]['c2'][burn_in_idxs[:,0]][None,:]
                    })                    
            else:
                for net in self.net_names:
                    batch.update({
                        # directly give hidden state
                        # Shape (1, batch_size, hidden_dim)
                        net+'_h':self.hidden_buf[net]['h'][idxs[:,0]][None,:],
                        net+'_c':self.hidden_buf[net]['c'][idxs[:,0]][None,:],
                        net+'_h2':self.hidden_buf[net]['h2'][idxs[:,0]][None,:],
                        net+'_c2':self.hidden_buf[net]['c2'][idxs[:,0]][None,:]
                        })
            batch = {k: torch.as_tensor(
                v, dtype=torch.float32).to(self.device) 
                if v is not None else v for k,v in batch.items()}
        else:
            idxs = np.random.randint(0, self.size, size=batch_size)
            # Shape is (batch_size, vector_dim)
            batch = dict(
                        obs=self.obs_buf[idxs],
                        obs2=self.obs2_buf[idxs],
                        act=self.act_buf[idxs],
                        rew=self.rew_buf[idxs],
                        done=self.done_buf[idxs],
                        h=None, 
                        c=None, 
                        h2=None,
                        c2=None)
            batch = {k: torch.as_tensor(
                v, dtype=torch.float32).to(self.device) 
                if v is not None else v for k,v in batch.items()}
        
        return batch



def td3(env_fn, actor_critic=core.ActorCritic, ac_kwargs=dict(), seed=0, 
        steps_per_epoch=4000, epochs=100, replay_size=int(1e6), gamma=0.99, 
        polyak=0.995, pi_lr=1e-3, q_lr=1e-3, batch_size=100, start_steps=10000, 
        update_after=1000, update_every=50, act_noise=0.1, target_noise=0.2, 
        noise_clip=0.5, policy_delay=2, num_test_episodes=10, max_ep_len=1000, 
        recurrent=True, hidden_dim=256, n_burn_in=40, n_sequence=80, n_overlap=40,
        device='cuda', logger_kwargs=dict(), save_freq=1,):
    """
    Twin Delayed Deep Deterministic Policy Gradient (TD3)


    Args:
        env_fn : A function which creates a copy of the environment.
            The environment must satisfy the OpenAI Gym API.

        actor_critic: The constructor method for a PyTorch Module with an ``act`` 
            method, a ``pi`` module, a ``q1`` module, and a ``q2`` module.
            The ``act`` method and ``pi`` module should accept batches of 
            observations as inputs, and ``q1`` and ``q2`` should accept a batch 
            of observations and a batch of actions as inputs. When called, 
            these should return:

            ===========  ================  ======================================
            Call         Output Shape      Description
            ===========  ================  ======================================
            ``act``      (batch, act_dim)  | Numpy array of actions for each 
                                           | observation.
            ``pi``       (batch, act_dim)  | Tensor containing actions from policy
                                           | given observations.
            ``q1``       (batch,)          | Tensor containing one current estimate
                                           | of Q* for the provided observations
                                           | and actions. (Critical: make sure to
                                           | flatten this!)
            ``q2``       (batch,)          | Tensor containing the other current 
                                           | estimate of Q* for the provided observations
                                           | and actions. (Critical: make sure to
                                           | flatten this!)
            ===========  ================  ======================================

        ac_kwargs (dict): Any kwargs appropriate for the ActorCritic object 
            you provided to TD3.

        seed (int): Seed for random number generators.

        steps_per_epoch (int): Number of steps of interaction (state-action pairs) 
            for the agent and the environment in each epoch.

        epochs (int): Number of epochs to run and train agent.

        replay_size (int): Maximum length of replay buffer.

        gamma (float): Discount factor. (Always between 0 and 1.)

        polyak (float): Interpolation factor in polyak averaging for target 
            networks. Target networks are updated towards main networks 
            according to:

            .. math:: \\theta_{\\text{targ}} \\leftarrow 
                \\rho \\theta_{\\text{targ}} + (1-\\rho) \\theta

            where :math:`\\rho` is polyak. (Always between 0 and 1, usually 
            close to 1.)

        pi_lr (float): Learning rate for policy.

        q_lr (float): Learning rate for Q-networks.

        batch_size (int): Minibatch size for SGD.

        start_steps (int): Number of steps for uniform-random action selection,
            before running real policy. Helps exploration.

        update_after (int): Number of env interactions to collect before
            starting to do gradient descent updates. Ensures replay buffer
            is full enough for useful updates.

        update_every (int): Number of env interactions that should elapse
            between gradient descent updates. Note: Regardless of how long 
            you wait between updates, the ratio of env steps to gradient steps 
            is locked to 1.

        act_noise (float): Stddev for Gaussian exploration noise added to 
            policy at training time. (At test time, no noise is added.)

        target_noise (float): Stddev for smoothing noise added to target 
            policy.

        noise_clip (float): Limit for absolute value of target policy 
            smoothing noise.

        policy_delay (int): Policy will only be updated once every 
            policy_delay times for each update of the Q-networks.

        num_test_episodes (int): Number of episodes to test the deterministic
            policy at the end of each epoch.

        max_ep_len (int): Maximum length of trajectory / episode / rollout.

        logger_kwargs (dict): Keyword args for EpochLogger.

        save_freq (int): How often (in terms of gap between epochs) to save
            the current policy and value function.

    """

    # record values to tensorboard when method log_tabular is called
    writer = SummaryWriter(log_dir=logger_kwargs['output_dir'])
    logger = EpochLogger(tb_writer=writer, **logger_kwargs)
    logger.save_config(locals())

    torch.manual_seed(seed)
    np.random.seed(seed)

    env, test_env = env_fn(), env_fn()
    obs_dim = env.observation_space.shape
    act_dim = env.action_space.shape[0]

    # Action limit for clamping: critically, assumes all dimensions share the same bound!
    act_limit = env.action_space.high[0]

    # Create actor-critic module and target networks
    ac = actor_critic(env.observation_space, env.action_space, 
                        recurrent, hidden_dim=hidden_dim, 
                        device=device, **ac_kwargs).to(device)
    ac_targ = deepcopy(ac).to(device)

    # Freeze target networks with respect to optimizers (only update via polyak averaging)
    for p in ac_targ.parameters():
        p.requires_grad = False
        
    # List of parameters for both Q-networks (save this for convenience)
    q_params = itertools.chain(ac.q1.parameters(), ac.q2.parameters())

    # Experience buffer
    model_names = ['pi', 'q1', 'q2']
    replay_buffer = ReplayBuffer(obs_dim, act_dim, replay_size, recurrent, 
                                 hidden_dim, n_sequence, n_burn_in, model_names, device)

    # Count variables (protip: try to get a feel for how different size networks behave!)
    var_counts = tuple(core.count_vars(module) for module in [ac.pi, ac.q1, ac.q2])
    logger.log('\nNumber of parameters: \t pi: %d, \t q1: %d, \t q2: %d\n'%var_counts)

    def burn_in(data):
        b_o, b_a, b_o2, b_a2 = data['b_obs'], data['b_act'],  data['b_obs2'], data['b_act2']
        pi_b_h, pi_b_c = data['pi_b_h'], data['pi_b_c']
        pi_b_h2, pi_b_c2 = data['pi_b_h2'], data['pi_b_c2']
        q1_b_h, q1_b_c = data['q1_b_h'], data['q1_b_c']
        q1_b_h2, q1_b_c2 = data['q1_b_h2'], data['q1_b_c2']
        q2_b_h, q2_b_c = data['q2_b_h'], data['q2_b_c']
        q2_b_h2, q2_b_c2 = data['q2_b_h2'], data['q2_b_c2']
            
        with torch.no_grad():
            # return last hidden state
            _, pi_hidden = ac.pi(b_o, (pi_b_h, pi_b_c))
            _, pi_targ_hidden = ac_targ.pi(b_o2, (pi_b_h2, pi_b_c2))
            pi_h, pi_c = pi_hidden
            pi_h2, pi_c2 = pi_targ_hidden
            
            _, q1_hidden = ac.q1(b_o, b_a, (q1_b_h, q1_b_c))
            _, q1_targ_hidden = ac.q1(b_o2, b_a2, (q1_b_h2, q1_b_c2))
            q1_h, q1_c = q1_hidden
            q1_h2, q1_c2 = q1_targ_hidden

            _, q2_hidden = ac.q2(b_o, b_a, (q2_b_h, q2_b_c))
            _, q2_targ_hidden = ac.q2(b_o2, b_a2, (q2_b_h2, q2_b_c2))
            q2_h, q2_c = q2_hidden
            q2_h2, q2_c2 = q2_targ_hidden

        data['pi_h'], data['pi_c'] = pi_h.detach(), pi_c.detach()
        data['pi_h2'], data['pi_c2'] = pi_h2.detach(), pi_c2.detach()
        data['q1_h'], data['q1_c'] = q1_h.detach(), q1_c.detach()
        data['q1_h2'], data['q1_c2'] = q1_h2.detach(), q1_c2.detach()
        data['q2_h'], data['q2_c'] = q2_h.detach(), q2_c.detach()
        data['q2_h2'], data['q2_c2'] = q2_h2.detach(), q2_c2.detach()

        return data

    # Set up function for computing TD3 Q-losses
    def compute_loss_q(data):
        o, a, r, o2, d = data['obs'], data['act'], data['rew'], data['obs2'], data['done']
        if recurrent:
            pi_h, pi_c, pi_h2, pi_c2 = data['pi_h'], data['pi_c'], data['pi_h2'], data['pi_c2']
            q1_h, q1_c, q1_h2, q1_c2 = data['q1_h'], data['q1_c'], data['q1_h2'], data['q1_c2']
            q2_h, q2_c, q2_h2, q2_c2 = data['q2_h'], data['q2_c'], data['q2_h2'], data['q2_c2']
        else:
            pi_h, pi_c, pi_h2, pi_c2 = None, None, None, None
            q1_h, q1_c, q1_h2, q1_c2 = None, None, None, None
            q2_h, q2_c, q2_h2, q2_c2 = None, None, None, None

        q1, _ = ac.q1(o, a, (q1_h, q1_c))
        q2, _ = ac.q2(o, a, (q2_h, q2_c))
        # Bellman backup for Q functions
        with torch.no_grad():
            pi_targ, _ = ac_targ.pi(o2, (pi_h2, pi_c2))

            # Target policy smoothing
            epsilon = torch.randn_like(pi_targ) * target_noise
            epsilon = torch.clamp(epsilon, -noise_clip, noise_clip)
            a2 = pi_targ + epsilon
            a2 = torch.clamp(a2, -act_limit, act_limit)

            # Target Q-values
            q1_pi_targ, _ = ac_targ.q1(o2, a2, (q1_h2, q1_c2))
            q2_pi_targ, _ = ac_targ.q2(o2, a2, (q2_h2, q2_c2))
            q_pi_targ = torch.min(q1_pi_targ, q2_pi_targ)
            backup = r + gamma * (1 - d) * q_pi_targ

        # MSE loss against Bellman backup
        loss_q1 = ((q1 - backup)**2).mean()
        loss_q2 = ((q2 - backup)**2).mean()
        loss_q = loss_q1 + loss_q2

        # Useful info for logging
        loss_info = dict(Q1Vals=q1.detach().cpu().numpy(),
                         Q2Vals=q2.detach().cpu().numpy())

        return loss_q, loss_info

    # Set up function for computing TD3 pi loss
    def compute_loss_pi(data):
        o = data['obs']
        if recurrent:
            pi_h, pi_c = data['pi_h'], data['pi_c']
            q1_h, q1_c = data['q1_h'], data['q1_c']
        else:
            pi_h, pi_c = None, None
            q1_h, q1_c = None, None

        a, _ = ac.pi(o, (pi_h, pi_c))
        q1_pi, _ = ac.q1(o, a, (q1_h, q1_c))
        return -q1_pi.mean()

    # Set up optimizers for policy and q-function
    pi_optimizer = Adam(ac.pi.parameters(), lr=pi_lr)
    q_optimizer = Adam(q_params, lr=q_lr)

    # Set up model saving
    logger.setup_pytorch_saver(ac)

    def update(data, timer):
        if recurrent and n_burn_in > 0:
            # hidden state after burn-in is added
            data = burn_in(data)

        # First run one gradient descent step for Q1 and Q2
        q_optimizer.zero_grad()
        loss_q, loss_info = compute_loss_q(data)
        loss_q.backward()
        q_optimizer.step()

        # Record things
        logger.store(LossQ=loss_q.item(), **loss_info)

        # Possibly update pi and target networks
        if timer % policy_delay == 0:

            # Freeze Q-networks so you don't waste computational effort 
            # computing gradients for them during the policy learning step.
            for p in q_params:
                p.requires_grad = False

            # Next run one gradient descent step for pi.
            pi_optimizer.zero_grad()
            loss_pi = compute_loss_pi(data)
            loss_pi.backward()
            pi_optimizer.step()

            # Unfreeze Q-networks so you can optimize it at next DDPG step.
            for p in q_params:
                p.requires_grad = True

            # Record things
            logger.store(LossPi=loss_pi.item())

            # Finally, update target networks by polyak averaging.
            with torch.no_grad():
                for p, p_targ in zip(ac.parameters(), ac_targ.parameters()):
                    # NB: We use an in-place operations "mul_", "add_" to update target
                    # params, as opposed to "mul" and "add", which would make new tensors.
                    p_targ.data.mul_(polyak)
                    p_targ.data.add_((1 - polyak) * p.data)

    def get_action(o, pi_hidden, noise_scale):
        # Shape (batch_size, step_length, vector_dim)
        o = torch.as_tensor(o, dtype=torch.float32).to(device).unsqueeze(0).unsqueeze(0)
        a, pi_hidden = ac.act(o, pi_hidden)
        a = a.squeeze(0).squeeze(0)
        a += noise_scale * np.random.randn(act_dim)
        # detach from current graph(memory exceeds limit if graph is not cut)
        if pi_hidden is not None:
            pi_hidden = (pi_hidden[0].detach(), pi_hidden[1].detach())
        return np.clip(a, -act_limit, act_limit), pi_hidden
    
    def get_q_hidden(o, a, q_net, q_hidden):
        # Shape (batch_size, step_length, vector_dim)
        o = torch.as_tensor(o, dtype=torch.float32).to(device).unsqueeze(0).unsqueeze(0)
        a = torch.as_tensor(a, dtype=torch.float32).to(device).unsqueeze(0).unsqueeze(0)
        _, next_q_hidden = q_net(o, a, q_hidden)
        # detach from current graph(memory exceeds limit if graph is not cut)
        if next_q_hidden is not None:
            next_q_hidden = (next_q_hidden[0].detach(), next_q_hidden[1].detach())
        return next_q_hidden

    def test_agent():
        for j in range(num_test_episodes):
            o, d, ep_ret, ep_len = test_env.reset(), False, 0, 0
            pi_hidden = ac.pi.get_initialized_hidden()
            while not(d or (ep_len == max_ep_len)):
                # Take deterministic actions at test time (noise_scale=0)
                a, pi_hidden = get_action(o, pi_hidden, 0)
                o, r, d, _ = test_env.step(a)
                ep_ret += r
                ep_len += 1
            logger.store(TestEpRet=ep_ret, TestEpLen=ep_len)

    # Prepare for interaction with environment
    total_steps = steps_per_epoch * epochs
    start_time = time.time()
    o, ep_ret, ep_len = env.reset(), 0, 0
    pi_hidden = ac.pi.get_initialized_hidden()
    q1_hidden = ac.q1.get_initialized_hidden()
    q2_hidden = ac.q2.get_initialized_hidden()

    # Main loop: collect experience in env and update/log each epoch
    for t in range(total_steps):
        
        # Until start_steps have elapsed, randomly sample actions
        # from a uniform distribution for better exploration. Afterwards, 
        # use the learned policy (with some noise, via act_noise). 
        if t > start_steps:
            a, next_pi_hidden = get_action(o, pi_hidden, act_noise)
            next_q1_hidden = get_q_hidden(o, a, ac.q1, q1_hidden)
            next_q2_hidden = get_q_hidden(o, a, ac.q1, q2_hidden)
        else:
            a = env.action_space.sample()
            _, next_pi_hidden = get_action(o, pi_hidden, act_noise)
            next_q1_hidden = get_q_hidden(o, a, ac.q1, q1_hidden)
            next_q2_hidden = get_q_hidden(o, a, ac.q2, q2_hidden)

        # Step the env
        o2, r, d, _ = env.step(a)
        ep_ret += r
        ep_len += 1

        # Ignore the "done" signal if it comes from hitting the time
        # horizon (that is, when it's an artificial terminal signal
        # that isn't based on the agent's state)
        d = False if ep_len==max_ep_len else d

        # Store experience to replay buffer
        hidden = {'pi':pi_hidden, 'q1':q1_hidden, 'q2':q2_hidden}
        next_hidden = {'pi':next_pi_hidden, 'q1':next_q1_hidden, 'q2':next_q2_hidden}
        replay_buffer.store(o, a, r, o2, d, hidden, next_hidden)

        # Super critical, easy to overlook step: make sure to update 
        # most recent observation!
        o = o2
        pi_hidden = next_pi_hidden
        q1_hidden = next_q1_hidden
        q2_hidden = next_q2_hidden

        # End of trajectory handling
        if d or (ep_len == max_ep_len):
            logger.store(EpRet=ep_ret, EpLen=ep_len)
            o, ep_ret, ep_len = env.reset(), 0, 0

        # Update handling
        if t >= update_after and t % update_every == 0:
            for j in range(update_every):
                batch = replay_buffer.sample_batch(batch_size)
                update(data=batch, timer=j)

        # End of epoch handling
        if (t+1) % steps_per_epoch == 0:
            epoch = (t+1) // steps_per_epoch

            # Save model
            if (epoch % save_freq == 0) or (epoch == epochs):
                logger.save_state({'env': env}, None)

            # Test the performance of the deterministic version of the agent.
            test_agent()

            # Log info about epoch
            logger.update_epoch(epoch)
            logger.log_tabular('EpRet', with_min_and_max=True)
            logger.log_tabular('TestEpRet', with_min_and_max=True)
            logger.log_tabular('EpLen', average_only=True)
            logger.log_tabular('TestEpLen', average_only=True)
            logger.log_tabular('TotalEnvInteracts', t)
            logger.log_tabular('Q1Vals', with_min_and_max=True)
            logger.log_tabular('Q2Vals', with_min_and_max=True)
            logger.log_tabular('LossPi', average_only=True)
            logger.log_tabular('LossQ', average_only=True)
            logger.log_tabular('Time', time.time()-start_time)
            logger.dump_tabular()
    writer.close()

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--env', type=str, default='HalfCheetah-v2')
    parser.add_argument('--hid', type=int, default=256)
    parser.add_argument('--gamma', type=float, default=0.99)
    parser.add_argument('--seed', '-s', type=int, default=0)
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--exp_name', type=str, default='td3')
    parser.add_argument('--recurrent', action='store_true')
    parser.add_argument('--batch_size', type=int, default=100)
    parser.add_argument('--n_sequence', type=int, default=80)
    parser.add_argument('--n_burn_in', type=int, default=40)
    parser.add_argument('--device', type=str, 
        default='cuda' if torch.cuda.is_available() else 'cpu')
    args = parser.parse_args()

    from rnnrl.utils.run_utils import setup_logger_kwargs
    logger_kwargs = setup_logger_kwargs(args.exp_name, args.seed)

    td3(lambda : PartialObservation(gym.make(args.env)), actor_critic=core.ActorCritic,
        ac_kwargs=dict(), 
        gamma=args.gamma, seed=args.seed, epochs=args.epochs,
        recurrent = args.recurrent, hidden_dim=args.hid, 
        device=args.device, logger_kwargs=logger_kwargs,
        batch_size=args.batch_size, n_sequence=args.n_sequence,
        n_burn_in=args.n_burn_in)
