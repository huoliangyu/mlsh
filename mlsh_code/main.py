import argparse
import tensorflow as tf
parser = argparse.ArgumentParser()
parser.add_argument('savename', type=str)
parser.add_argument('--task', type=str)
parser.add_argument('--num_subs', type=int)
parser.add_argument('--macro_duration', type=int)
parser.add_argument('--num_rollouts', type=int)
parser.add_argument('--warmup_time', type=int)
parser.add_argument('--train_time', type=int)
parser.add_argument('--force_subpolicy', type=int)
parser.add_argument('--replay', type=str)
parser.add_argument('-s', action='store_true')
parser.add_argument('--continue_iter', type=str)
parser.add_argument('--restore',default = "False",type=str)
parser.add_argument('--seed', default = 1401,type=int)
args = parser.parse_args()

# python main.py --task MovementBandits-v0 --num_subs 2 --macro_duration 10 --num_rollouts 1000 --warmup_time 60 --train_time 1 --replay True test

from mpi4py import MPI
from rl_algs.common import set_global_seeds, tf_util as U
import os.path as osp
import gym, logging
import numpy as np
from collections import deque
from gym import spaces
import misc_util
import sys
import shutil
import subprocess
import master

def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

replay = str2bool(args.replay)
args.replay = str2bool(args.replay)

restore = str2bool(args.restore)
args.restore = str2bool(args.restore)
RELPATH = osp.join(args.savename)
LOGDIR = osp.join('/root/results' if sys.platform.startswith('linux') else '/tmp', RELPATH)

# def callback(it):
#     if MPI.COMM_WORLD.Get_rank()==0:
#         if it % 5 == 0 and it > 3 and not replay:
#             fname = osp.join("savedir/", 'checkpoints', '%.5i'%it)
#             U.save_state(fname)
#     if it == 0 and args.continue_iter is not None:
#         fname = osp.join("savedir/"+args.savename+"/checkpoints/", str(args.continue_iter))
#         U.load_state(fname)
#         pass
def callback(it,savename =args.savename, var_list=None,restore=False,savedir = "./savedir",save=True,continue_iter = 0,force_save=False):
    if MPI.COMM_WORLD.Get_rank()==0:
        # # if it % 5 == 0 and it > 3 and not replay:
        # if not replay:
        #     fname = osp.join("savedir/", 'checkpoints', '%.5i'%it)
        #     U.save_state(fname)
        # if it == 0 and args.continue_iter is not None:
        #     fname = osp.join("savedir/"+args.savename+"/checkpoints/", str(args.continue_iter))
        if force_save:
            fname = osp.join("{}/{}".format(savedir,savename), 'checkpoints', '%.5i'%it)
            U.save_state(fname,var_list) 
        elif it % 20 == 0 and it > 20 and not replay and save and not restore:
            fname = osp.join("{}/{}".format(savedir,savename), 'checkpoints', '%.5i'%it)
            U.save_state(fname,var_list)
    if it == 0 and restore:
        fname = osp.join("{}/{}".format(savedir,savename)+"/checkpoints/", str(continue_iter))
        U.load_state(fname)


def train():
    num_timesteps=1e9
    seed = args.seed
    # seed = 1401
    rank = MPI.COMM_WORLD.Get_rank()
    sess = U.single_threaded_session()
    sess.__enter__()
    workerseed = seed + 1000 * MPI.COMM_WORLD.Get_rank()
    rank = MPI.COMM_WORLD.Get_rank()
    set_global_seeds(workerseed)

    # if rank != 0:
    #     logger.set_level(logger.DISABLED)
    # logger.log("rank %i" % MPI.COMM_WORLD.Get_rank())

    world_group = MPI.COMM_WORLD.Get_group()
    mygroup = rank % 10
    theta_group = world_group.Incl([x for x in range(MPI.COMM_WORLD.size) if (x % 10 == mygroup)])
    comm = MPI.COMM_WORLD.Create(theta_group)
    comm.Barrier()
    # comm = MPI.COMM_WORLD

    master.start(callback, args=args, workerseed=workerseed, rank=rank, comm=comm)

def main():
    if MPI.COMM_WORLD.Get_rank() == 0 and osp.exists(LOGDIR):
        shutil.rmtree(LOGDIR)
    MPI.COMM_WORLD.Barrier()
    # with logger.session(dir=LOGDIR):
    train()

if __name__ == '__main__':
    main()
