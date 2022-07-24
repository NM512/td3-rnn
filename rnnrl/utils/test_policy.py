import time
import joblib
import os
import os.path as osp
import copy
import torch
from rnnrl import EpochLogger


def load_policy_and_env(fpath, itr='last', deterministic=False, device='cuda'):
    """
    Load a policy from save along with RL env.

    Not exceptionally future-proof, but it will suffice for basic uses of the 
    implementations.
    """

    # handle which epoch to load from
    if itr=='last':
        # check filenames for epoch (AKA iteration) numbers, find maximum value        
        pytsave_path = osp.join(fpath, 'pyt_save')
        # Each file in this folder has naming convention 'modelXX.pt', where
        # 'XX' is either an integer or empty string. Empty string case
        # corresponds to len(x)==8, hence that case is excluded.
        saves = [int(x.split('.')[0][5:]) for x in os.listdir(pytsave_path) if len(x)>8 and 'model' in x]

        itr = '%d'%max(saves) if len(saves) > 0 else ''

    else:
        assert isinstance(itr, int), \
            "Bad value provided for itr (needs to be int or 'last')."
        itr = '%d'%itr

    # load the get_action function
    get_action, init_hidden = load_pytorch_policy(fpath, itr, deterministic, device)

    # try to load environment from save
    # (sometimes this will fail because the environment could not be pickled)
    try:
        state = joblib.load(osp.join(fpath, 'vars'+itr+'.pkl'))
        env = state['env']
    except:
        env = None

    return env, get_action, init_hidden

def load_pytorch_policy(fpath, itr, deterministic=False, device='cuda'):
    """ Load a pytorch policy saved with Logger."""
    
    fname = osp.join(fpath, 'pyt_save', 'model'+itr+'.pt')
    print('\n\nLoading from %s.\n\n'%fname)

    model = torch.load(fname).to(device)

    # make function for producing an action given a single state
    def get_action(x, hidden):
        hidden = (hidden[0], init_hidden[1])

        with torch.no_grad():
            x = torch.as_tensor(x, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(device)
            action, hidden = model.act(x, hidden)
            action = action.squeeze(0).squeeze(0)
        if hidden is not None:
            hidden = (hidden[0].detach(), hidden[1].detach())
        return action, hidden
    init_hidden = model.pi.get_initialized_hidden()
    
    return get_action, init_hidden


def run_policy(env, get_action, init_hidden, max_ep_len=None, num_episodes=100, render=True):

    assert env is not None, \
        "Environment not found!\n\n It looks like the environment wasn't saved, " + \
        "and we can't run the agent in it. :( \n\n Check out the readthedocs " + \
        "page on Experiment Outputs for how to handle this situation."

    logger = EpochLogger()
    o, r, d, ep_ret, ep_len, n = env.reset(), 0, False, 0, 0, 0
    hidden = copy.deepcopy(init_hidden)
    while n < num_episodes:
        if render:
            env.render()
            time.sleep(1e-3)

        a, hidden = get_action(o, hidden)
        o, r, d, _ = env.step(a)
        ep_ret += r
        ep_len += 1

        if d or (ep_len == max_ep_len):
            logger.store(EpRet=ep_ret, EpLen=ep_len)
            print('Episode %d \t EpRet %.3f \t EpLen %d'%(n, ep_ret, ep_len))
            o, r, d, ep_ret, ep_len = env.reset(), 0, False, 0, 0
            hidden = copy.deepcopy(init_hidden)
            n += 1

    logger.log_tabular('EpRet', with_min_and_max=True)
    logger.log_tabular('EpLen', average_only=True)
    logger.dump_tabular()


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('fpath', type=str)
    parser.add_argument('--len', '-l', type=int, default=0)
    parser.add_argument('--episodes', '-n', type=int, default=100)
    parser.add_argument('--norender', '-nr', action='store_true')
    parser.add_argument('--itr', '-i', type=int, default=-1)
    parser.add_argument('--deterministic', '-d', action='store_true')
    parser.add_argument('--device', type=str, default='cuda')
    
    args = parser.parse_args()
    env, get_action, init_hidden = load_policy_and_env(args.fpath, 
                                          args.itr if args.itr >=0 else 'last',
                                          args.deterministic, args.device)
    run_policy(env, get_action, init_hidden, args.len, args.episodes, not(args.norender))