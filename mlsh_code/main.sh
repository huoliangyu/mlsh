export PYTHONPATH=$PYTHONPATH:/home/hly/mlsh_aaai/mlsh/gym
export PYTHONPATH=$PYTHONPATH:/home/hly/mlsh_aaai/mlsh/rl-algs
# /home/hly/anaconda3/envs/mlsh_aaai/bin/python  main.py --task AntBandits-v1 --num_subs 2 --macro_duration 1000 --num_rollouts 2000 --warmup_time 20 --train_time 30 --replay False AntAgent
/home/hly/anaconda3/envs/mlsh_aaai/bin/python  main.py --task MovementBandits-v0 --num_subs 2 --macro_duration 10 --num_rollouts 2000 --warmup_time 9 --train_time 1 --replay False MovementBandits

