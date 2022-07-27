# rnnrl
Pytorch implementations of RL (Reinforcement Learning) algorithms with RNN (Reccurent Neural Network)

Disclaimer: My code is very much based on TD3 of [openAI/spinningup](https://github.com/openai/spinningup) and motivated from [AntoineTheb's work](https://github.com/AntoineTheb/RNN-RL)

# Motivations
This repository is intended to test Actor-Critic with RNN to better understand what happens when applied to POMDP (Partially Observable Markov Decision Process) tasks.
The naive technique to mitigate POMDP is to simply use stacked observations as observation at some point.


[DeepMind/R2D2](https://www.deepmind.com/publications/recurrent-experience-replay-in-distributed-reinforcement-learning) incporporated RNN into distributed reinforcement learning to achieve significant performance improvements on Atari tasks.
In general, training RL algorithm with RNN is simple and easy if we only use on-policy algorithm and have no replay buffer for that.
Because it's not necessary to consider how to restore the hidden state which has to be produced from current network if replay buffer is not used.
However it's known that utilizing off-policy RL algorithms with replay buffer makes training more efficient.

There are following techniques proposed to adapt off-policy and replay buffer to actor-critic algorithm.
- 'Stored state' keep the hidden state of the RNN associated with each timestep when rolling out.
- Allow network a "Burn-in" period by just letting network go through state before training timestep.

TD3 which is off-policy and has replay buffer is used for following benchmarks.

# What I want to answer for
- Difference between the network using simple stacked observation and the network with RNN
- How following techniques make difference
  - Preserved hidden state
  - Burn-in process
- How parameters within above techniques is critical for performance


# How to install

```
pip install -e .
```

# How to run
without RNN, using CPU
```
python rnnrl/algos/pytorch/td3/td3.py --env Pendulum-v0 --seed=$i --device cpu --batch_size 16 --hid 32
```


with RNN, using GPU

```
python rnnrl/algos/pytorch/td3/td3.py --env Pendulum-v0 --seed=$i --device cuda --batch_size 16 --hid 32 --recurrent --stored_state --n_burn_in 10 --n_sequence 20
```



# Results
Benchmarks are executed under environment of **Pendulum-v0** with [PartialObservation](https://github.com/m-naoki/rnnrl/blob/6d3a58d728b30b8f122003bdb54c11ccda8e45e2/rnnrl/utils/wrappers.py#L6).

[PartialObservation](https://github.com/m-naoki/rnnrl/blob/6d3a58d728b30b8f122003bdb54c11ccda8e45e2/rnnrl/utils/wrappers.py#L6) is a wrapper to allow policy to receive observation only once in 3 times of steps for making POMDP.

![](/plots/1.png)

![](/plots/2.png)

**To be continued...**
