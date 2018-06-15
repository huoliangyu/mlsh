import gym
import test_envs
import tensorflow as tf
import rollouts
from policy_network import Policy
from subpolicy_network import SubPolicy
from observation_network import Features
from learner import Learner
import rl_algs.common.tf_util as U
import numpy as np
# from tinkerbell import logger
import pickle
import logger
from console_util import fmt_row
import os
import shutil
import time
from collections import deque
def start(callback, args, workerseed, rank, comm):
    env = gym.make(args.task)
    env.seed(workerseed)
    np.random.seed(workerseed)
    ob_space = env.observation_space
    ac_space = env.action_space

    num_subs = args.num_subs
    macro_duration = args.macro_duration
    num_rollouts = args.num_rollouts
    warmup_time = args.warmup_time
    train_time = args.train_time
    test_time = 1
    test_steps = warmup_time # warmup time or warmup time+train_time
    num_batches = 15
    index = 1
    savename = "env_{}_subs_{}_warmup_{}_train_{}_T_{}_index_{}".format(args.task,num_subs,args.warmup_time,args.train_time,args.macro_duration, index)
    logdir = "./savedir/{}".format(savename)
    if os.path.exists(logdir):
        shutil.rmtree(logdir)
    os.makedirs(logdir)
    # num_batches = 1000
    # observation in.
    ob = U.get_placeholder(name="ob", dtype=tf.float32, shape=[None, ob_space.shape[0]])
    # ob = U.get_placeholder(name="ob", dtype=tf.float32, shape=[None, 104])

    # features = Features(name="features", ob=ob)
    policy = Policy(name="policy", ob=ob, ac_space=ac_space, hid_size=32, num_hid_layers=2, num_subpolicies=num_subs)
    old_policy = Policy(name="old_policy", ob=ob, ac_space=ac_space, hid_size=32, num_hid_layers=2, num_subpolicies=num_subs)

    sub_policies = [SubPolicy(name="sub_policy_%i" % x, ob=ob, ac_space=ac_space, hid_size=64, num_hid_layers=2) for x in range(num_subs)]
    old_sub_policies = [SubPolicy(name="old_sub_policy_%i" % x, ob=ob, ac_space=ac_space, hid_size=64, num_hid_layers=2) for x in range(num_subs)]

    # learner = Learner(env, policy, old_policy, sub_policies, old_sub_policies, comm, savename,logdir, clip_param=0.2, entcoeff=0, optim_epochs=10, optim_stepsize=3e-4, optim_batchsize=1000)
    learner = Learner(env, policy, old_policy, sub_policies, old_sub_policies, comm, savename,logdir,clip_param=0.2, entcoeff=0, optim_epochs=10, optim_stepsize=3e-5, optim_batchsize=64)
    
    # rollout = rollouts.traj_segment_generator(policy, sub_policies, env, macro_duration, num_rollouts, stochastic=True, test_steps=warmup_time+train_time, args=args)
    rollout = rollouts.traj_segment_generator(policy, sub_policies, env, macro_duration, num_rollouts, stochastic=True, test_steps=test_steps, total_steps=warmup_time+train_time, args=args)
    
    logger_loss_name = ["mini_ep","glo_rew","loc_rew","sub_rate","time","test"]
    start_time = time.time()
    real_goal = 0
    gmean_final = 0
    sub_rate_final = None
    total_rewbuffer = deque()
    test_final = 0
    hightest = -1000
    for x in range(10000):
        # callback(x,savename)
        if x == 0:
            learner.syncSubpolicies()
            print("synced subpols")
        # Run the inner meta-episode.

        policy.reset()
        learner.syncMasterPolicies()

        env.env.randomizeCorrect()
        shared_goal = comm.bcast(env.env.realgoal, root=0)
        real_goal =env.env.realgoal = shared_goal

        # print("It is iteration %d so i'm changing the goal to %s" % (x, env.env.realgoal))
        logger.log("It is iteration %d so i'm changing the goal to %s" % (x, env.env.realgoal))

        mini_ep = 0 if x > 0 else -1 * (rank % 10)*int(warmup_time+train_time / 10)
        # mini_ep = 0

        totalmeans = []
        logger.log(fmt_row(10, logger_loss_name))
        while mini_ep < warmup_time+train_time+test_time:
            mini_ep += 1
            
            running_time = time.time()-start_time
            start_time = time.time()
            # rollout
            rolls = rollout.__next__()
            allrolls = []
            allrolls.append(rolls)
            if mini_ep==test_steps+test_time:
                is_test = True
                
            else:
                is_test = False
            # train theta
            rollouts.add_advantage_macro(rolls, macro_duration, 0.99, 0.98)
            gmean, lmean,sub_rate = learner.updateMasterPolicy(rolls,is_test)
            # train phi
            test_seg = rollouts.prepare_allrolls(allrolls, macro_duration, 0.99, 0.98, num_subpolicies=num_subs)
            learner.updateSubPolicies(test_seg, num_batches, (mini_ep >= warmup_time)and(is_test is False))
            if is_test:
                if gmean>hightest:
                    callback(x,savename)
                    
            gmean_final=gmean
            sub_rate_final =sub_rate
            
            # learner.updateSubPolicies(test_seg,
            # log
            # print(("%d: global: %s, local: %s" % (mini_ep, gmean, lmean)))
             
            test_flag = hightest if is_test else 0
            logger_list =[mini_ep,gmean,lmean,sub_rate ,running_time,test_flag]
            logger.log(fmt_row(10, logger_list))
            if args.s:
                totalmeans.append(gmean)
                with open('outfile'+str(x)+'.pickle', 'wb') as fp:
                    pickle.dump(totalmeans, fp)
        total_rewbuffer.append(gmean_final)
        total_rew = np.mean(total_rewbuffer)
        # learner.add_total_info(x,real_goal,total_rew,gmean_final,sub_rate_final)