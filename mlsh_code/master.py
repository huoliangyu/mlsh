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

is_restore = False
is_save = True
is_record_each_iter = True
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
    # test_steps = warmup_time # warmup time or warmup time+train_time
    num_batches = 15
    import dash_meta
    index = 3 # 2 represent no macrotime,3 represent use last_down_time
    
    saveinfo = "NormalRebffing_and_record_each_iter".format(dash_meta.index_num)
    # saveinfo = "newtask"
    savename = "env_{}_subs_{}_warmup_{}_train_{}_T_{}_info_{}_index_{}".format(args.task,num_subs,args.warmup_time,args.train_time,args.macro_duration, saveinfo,index)
    logdir = "./savedir/{}".format(savename)
    if os.path.exists(logdir):
        shutil.rmtree(logdir)
    os.makedirs(logdir)
    # num_batches = 1000
    # observation in.
    ob = U.get_placeholder(name="ob", dtype=tf.float32, shape=[None, ob_space.shape[0]])
    # ob = U.get_placeholder(name="ob", dtype=tf.float32, shape=[None, 104])

    # features = Features(name="features", ob=ob)
    policy = Policy(name="policy", ob=ob, ac_space=ac_space, hid_size=64, num_hid_layers=2, num_subpolicies=num_subs)
    old_policy = Policy(name="old_policy", ob=ob, ac_space=ac_space, hid_size=64, num_hid_layers=2, num_subpolicies=num_subs)

    sub_policies = [SubPolicy(name="sub_policy_%i" % x, ob=ob, ac_space=ac_space, hid_size=64, num_hid_layers=2) for x in range(num_subs)]
    old_sub_policies = [SubPolicy(name="old_sub_policy_%i" % x, ob=ob, ac_space=ac_space, hid_size=64, num_hid_layers=2) for x in range(num_subs)]

    # learner = Learner(env, policy, old_policy, sub_policies, old_sub_policies, comm, savename,logdir, clip_param=0.2, entcoeff=0, optim_epochs=10, optim_stepsize=3e-4, optim_batchsize=1000)
    learner = Learner(env, policy, old_policy, sub_policies, old_sub_policies, comm, savename,logdir,clip_param=0.2, entcoeff=0, optim_epochs=10, optim_stepsize=3e-5, optim_batchsize=64)
    
    # rollout = rollouts.traj_segment_generator(policy, sub_policies, env, macro_duration, num_rollouts, stochastic=True, test_steps=warmup_time+train_time, args=args)
    # rollout = rollouts.traj_segment_generator(policy, sub_policies, env, macro_duration, num_rollouts, stochastic=True, test_steps=test_steps, total_steps=warmup_time+train_time, args=args)
    rollout = rollouts.traj_segment_generator(policy, sub_policies, env, macro_duration, num_rollouts, stochastic=True, test_steps=warmup_time+train_time, args=args)
    
    logger_loss_name = ["mini_ep","glo_rew","loc_rew","sub_rate","time","test"]
    start_time = time.time()
    import dash_meta
    num_task = dash_meta.index_num
    real_goal = 0
    gmean_final = 0
    sub_rate_final = None
    total_rewbuffer = [deque() for _ in range(num_task)]
    hightest = [-1000 for _ in range(num_task)]

    test_final = 0
    ssim_final = 0
    jitter_final = 0
    rebuffing_final = 0
    freeze_final = 0
    creal_final = 0
    buffer_final = 0
    sub_num_rate_final = None

    rewbuffer = [deque(maxlen=100) for _ in range(num_task)]
    ssimbuffer = [deque(maxlen=100) for _ in range(num_task)]
    jitterbuffer = [deque(maxlen=100) for _ in range(num_task)]
    rebuffing_timebuffer = [deque(maxlen=100) for _ in range(num_task)]
    freezebuffer = [deque(maxlen=100) for _ in range(num_task)]
    crealbuffer =[deque(maxlen=100) for _ in range(num_task)] 
    bufferbuffer = [deque(maxlen=100) for _ in range(num_task)] 
    if is_restore:
        
        restore_name_list = []# sub-3 in the first and 2nd and 1st
        if False:
            for i in range(num_subs):
                
                restore_name = restore_name_list[i]
                varlist = sub_policies[0].get_trainable_variables()
                callback(0,restore_name,var_list=varlist,restore=True,savedir=logdir,save=is_save)
                if i == num_subs-1:
                    break
                assign_old_eq_new = U.function([],[], updates=[tf.assign(oldv, newv)
                for (oldv, newv) in zip(sub_policies[num_subs-i-1].get_variables(), sub_policies[0].get_variables())])
                assign_old_eq_new()
        else:
            restore_name = "env_DashMeta-v0_subs_3_warmup_20_train_40_T_20_info_ToSaveModel_index_3"
            continue_iter = '03208'
            for i in range(num_subs):
                varlist = sub_policies[i].get_trainable_variables()
                callback(0,restore_name,var_list=varlist,restore=True,save=is_save,continue_iter=continue_iter)
            print ("restore model over in {}".format(restore_name))

    for x in range(2000):
        # callback(x,savename,savedir=logdir,save=is_save)
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
            if mini_ep==warmup_time+train_time+test_time:
                is_test = True
                
            else:
                is_test = False
            # train theta
            rollouts.add_advantage_macro(rolls, macro_duration, 0.99, 0.98)
            gmean, lmean,sub_rate,ssim,jitter,rebuffing,freeze,creal,buffer = learner.updateMasterPolicy(rolls,is_test)
            # train phi
            test_seg = rollouts.prepare_allrolls(allrolls, macro_duration, 0.99, 0.98, num_subpolicies=num_subs)
            learner.updateSubPolicies(test_seg, num_batches, (mini_ep >= warmup_time)and(is_test is False))
            if is_test:
                test_final = gmean
                ssim_final = ssim
                jitter_final = jitter
                rebuffing_final = rebuffing
                freeze_final = freeze
                creal_final = creal
                buffer_final = buffer
                sub_num_rate_final = sub_rate
                if real_goal >= len(hightest):
                    raise Exception("real is {} but hight is {}".format(real_goal,hightest))
                else:
                    if gmean>hightest[real_goal]:
                        callback(x,savename,save=is_save)
                        hightest[real_goal] = gmean
                    
            gmean_final=gmean
            sub_rate_final =sub_rate
            
            # learner.updateSubPolicies(test_seg,
            # log
            # print(("%d: global: %s, local: %s" % (mini_ep, gmean, lmean)))
             
            test_flag = hightest[real_goal] if is_test else 0
            logger_list =[mini_ep,gmean,lmean,sub_rate ,running_time,test_flag]
            logger.log(fmt_row(10, logger_list))
            if args.s:
                totalmeans.append(gmean)
                with open('outfile'+str(x)+'.pickle', 'wb') as fp:
                    pickle.dump(totalmeans, fp)
            if is_record_each_iter:
                total_rewbuffer[real_goal].append(gmean)
                total_rew = np.mean(total_rewbuffer[real_goal])

                rewbuffer[real_goal].append(gmean)
                ssimbuffer[real_goal].append(ssim)
                jitterbuffer[real_goal].append(jitter)
                rebuffing_timebuffer[real_goal].append(rebuffing)
                freezebuffer[real_goal].append(freeze)
                crealbuffer[real_goal].append(creal_final)
                bufferbuffer[real_goal].append(buffer)
            
                learner.add_total_info(x*(warmup_time+train_time+test_time)+mini_ep-1,real_goal,total_rew,gmean,sub_rate,np.mean(rewbuffer[real_goal]),np.mean(ssimbuffer[real_goal]),np.mean(jitterbuffer[real_goal]),np.mean(rebuffing_timebuffer[real_goal]),np.mean(freezebuffer[real_goal]),np.mean(crealbuffer[real_goal]),np.mean(bufferbuffer[real_goal]),sub_rate)
        if not is_record_each_iter: 
            total_rewbuffer[real_goal].append(gmean_final)
            total_rew = np.mean(total_rewbuffer[real_goal])

            rewbuffer[real_goal].append(test_final)
            ssimbuffer[real_goal].append(ssim_final)
            jitterbuffer[real_goal].append(jitter_final)
            rebuffing_timebuffer[real_goal].append(rebuffing_final)
            freezebuffer[real_goal].append(freeze_final)
            crealbuffer[real_goal].append(creal)
            bufferbuffer[real_goal].append(buffer_final)
            
            learner.add_total_info(x,real_goal,total_rew,gmean_final,sub_rate_final,np.mean(rewbuffer[real_goal]),np.mean(ssimbuffer[real_goal]),np.mean(jitterbuffer[real_goal]),np.mean(rebuffing_timebuffer[real_goal]),np.mean(freezebuffer[real_goal]),np.mean(crealbuffer[real_goal]),np.mean(bufferbuffer[real_goal]),sub_num_rate_final)
