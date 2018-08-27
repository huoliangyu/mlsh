import gym
import test_envs
import tensorflow as tf
import rollouts
sec_der_weight = 1
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

is_restore = False

is_save = True

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

    num_batches = 15

    index = 1
    saveinfo = "sec_der"
    savename = "env_{}_subs_{}_warmup_{}_train_{}_T_{}_weight_{}_info_{}_index_{}".format(args.task,num_subs,args.warmup_time,args.train_time,args.macro_duration, sec_der_weight,saveinfo,index)
    logdir = "./savedir/{}".format(savename)
    
    if os.path.exists(logdir):
        shutil.rmtree(logdir)
    os.makedirs(logdir)
    
    # observation in.
    ob = U.get_placeholder(name="ob", dtype=tf.float32, shape=[None, ob_space.shape[0]])
    # ob = U.get_placeholder(name="ob", dtype=tf.float32, shape=[None, 104])

    # features = Features(name="features", ob=ob)
    policy = Policy(name="policy", ob=ob, ac_space=ac_space, hid_size=32, num_hid_layers=2, num_subpolicies=num_subs)
    old_policy = Policy(name="old_policy", ob=ob, ac_space=ac_space, hid_size=32, num_hid_layers=2, num_subpolicies=num_subs)

    sub_policies = [SubPolicy(name="sub_policy_%i" % x, ob=ob, ac_space=ac_space, hid_size=32, num_hid_layers=2) for x in range(num_subs)]
    old_sub_policies = [SubPolicy(name="old_sub_policy_%i" % x, ob=ob, ac_space=ac_space, hid_size=32, num_hid_layers=2) for x in range(num_subs)]

    learner = Learner(env, policy, old_policy, sub_policies, old_sub_policies, comm, savename,logdir,clip_param=0.2, entcoeff=0, optim_epochs=10, optim_stepsize=3e-5, optim_batchsize=64)
    rollout = rollouts.traj_segment_generator(policy, sub_policies, env, macro_duration, num_rollouts, stochastic=True, args=args)

    logger_loss_name = ["mini_ep","glo_rew","loc_rew","glo_dcos","loc_dcos","sub_rate","time"]
    start_time = time.time()
    num_task = num_subs
    real_goal = 0
    total_rewbuffer = [deque() for _ in range(num_task)]
    total_dcosbuffer = [deque() for _ in range(num_task)]
    if is_restore:
        # restore_name_list = []# sub-3 in the first and 2nd and 1st
        restore_name = "env_MovementBandits-v0_subs_2_warmup_9_train_1_T_10_weight_0_info_sec_der_index_1"
        continue_iter = '00719'
        for i in range(num_subs):
            varlist = sub_policies[i].get_trainable_variables()
            callback(0,restore_name,var_list=varlist,restore=True,save=is_save,continue_iter=continue_iter)
        print ("restore model over in {}".format(restore_name))





    for x in range(10000):
        # callback(x)
        if x == 0:
            learner.syncSubpolicies()
            print("synced subpols")
        # Run the inner meta-episode.

        policy.reset()
        learner.syncMasterPolicies()

        env.env.randomizeCorrect()
        shared_goal = comm.bcast(env.env.realgoal, root=0)
        real_goal =env.env.realgoal = shared_goal
        if type(real_goal)==int:
            pass
        else:
            if real_goal.shape[0]==2:
                if real_goal[0]==0:
                    real_goal = 0
                else:
                    real_goal = 1

        # print("It is iteration %d so i'm changing the goal to %s" % (x, env.env.realgoal))
        logger.log("It is iteration %d so i'm changing the goal to %s" % (x, env.env.realgoal))
        logger.log(fmt_row(10, logger_loss_name))
        mini_ep = 0 if x > 0 else -1 * (rank % 10)*int(warmup_time+train_time / 10)
        # mini_ep = 0

        totalmeans = []
        while mini_ep < warmup_time+train_time:
            mini_ep += 1
            
            running_time = time.time()-start_time
            start_time = time.time()
            # rollout
            rolls = rollout.__next__()
            allrolls = []
            allrolls.append(rolls)
            # train theta
            rollouts.add_advantage_macro(rolls, macro_duration, 0.99, 0.98)
            gmean, lmean,sub_rate = learner.updateMasterPolicy(rolls)
            # train phi
            test_seg = rollouts.prepare_allrolls(allrolls, macro_duration, 0.99, 0.98, num_subpolicies=num_subs)
            gdcos,ldcos,is_nan = learner.updateSubPolicies(test_seg, num_batches, (mini_ep >= warmup_time))
            if is_nan:
                callback(x,savename,save=True,force_save=True)
                raise Exception("dcos nan:", gdcos)
            # learner.updateSubPolicies(test_seg,
            # log
            # print(("%d: global: %s, local: %s" % (mini_ep, gmean, lmean)))
            
            logger_list =[mini_ep,gmean,lmean,gdcos,ldcos,sub_rate ,running_time]
            logger.log(fmt_row(10, logger_list))

            if args.s:
                totalmeans.append(gmean)
                with open('outfile'+str(x)+'.pickle', 'wb') as fp:
                    pickle.dump(totalmeans, fp)
            if mini_ep>=warmup_time+train_time:
                total_rewbuffer[real_goal].append(gmean)
                total_rew = np.mean(total_rewbuffer[real_goal])
                total_dcosbuffer[real_goal].append(gdcos)
                total_dcos= np.mean(total_dcosbuffer[real_goal])
                # learner.add_total_info(x*(warmup_time+train_time)+mini_ep-1,real_goal,total_rew,gmean,total_dcos,gdcos,sub_rate)
                learner.add_total_info(x*train_time+mini_ep-1-warmup_time,real_goal,total_rew,gmean,total_dcos,gdcos,sub_rate)
