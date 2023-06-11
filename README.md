# RNN-RL
Pytorch implementations of RL (Reinforcement Learning) algorithms with RNN (Reccurent Neural Network) and Experience Replay

Disclaimer: My code is based on TD3 of [openAI/spinningup](https://github.com/openai/spinningup).

# Motivations
Experiment RL containing RNN and Experience Replay to better understand how following techniches and parameters affect.

[R2D2](https://www.deepmind.com/publications/recurrent-experience-replay-in-distributed-reinforcement-learning) incporporated RNN into distributed reinforcement learning to achieve significant performance improvements on Atari tasks.

In that paper, they investigated the training of RNNs with Experience Replay. And proposed following techniques adapt off-policy and Experience Replay to Actor-Critic algorithm.
- 'Stored state' keep the hidden state of to the experience buffer when roll out.
- "Burn-in" allow network to go through state before training timestep.

TD3 which is Actor-Critic algorithm which has replay buffer is used for following benchmarks.

# What I want to answer for
- Difference using simple stacked observation and RNN network against POMDP task
- How following techniques make difference
  - Stored state
  - Burn-in process
- How parameters of above techniques affect performance


# How to install

```
pip install -e .
```

# How to run
without RNN, using CPU
```
python rnnrl/algos/pytorch/td3/td3.py --env Pendulum-v0 --seed=$i --device cpu
```


with RNN, using GPU

```
python rnnrl/algos/pytorch/td3/td3.py --env Pendulum-v0 --seed=$i --device cuda --recurrent
```

# Results

Benchmarks are executed under environment of **Pendulum-v0** with [PartialObservation](https://github.com/m-naoki/rnnrl/blob/6d3a58d728b30b8f122003bdb54c11ccda8e45e2/rnnrl/utils/wrappers.py#L6).

[PartialObservation](https://github.com/m-naoki/rnnrl/blob/6d3a58d728b30b8f122003bdb54c11ccda8e45e2/rnnrl/utils/wrappers.py#L6) is a wrapper to allow policy to receive observation only once in 3 times of steps for making POMDP.
The naive technique to mitigate POMDP is to simply use stacked observations as observation at some point.

![1](https://github.com/NM512/td3-rnn/assets/70328564/43fda1f3-26ad-4ef5-910b-aca292ab42eb)

![2](https://github.com/NM512/td3-rnn/assets/70328564/d2a9597e-5afd-4301-aa38-86d4ffb66565)
