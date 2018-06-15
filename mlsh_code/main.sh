export PYTHONPATH=$PYTHONPATH:/home/hly/dash/mlsh/gym
export PYTHONPATH=$PYTHONPATH:/home/hly/dash/mlsh/rl-algs
 /home/hly/anaconda3/envs/dash_meta/bin/python main.py --task MovementBandits-v0 --num_subs 2 --macro_duration 10 --num_rollouts 2000 --warmup_time 9 --train_time 1 --replay False movementbanditsAgent
# /home/hly/anaconda3/envs/dash_meta/bin/python main.py --task DashMeta-v0 --num_subs  3 --macro_duration 10 --num_rollouts 1000 --warmup_time 20 --train_time 40 --replay False DashMetaAgent
# /home/hly/anaconda3/envs/dash_meta/bin/python main.py --task DashMeta-v0 --num_subs  1 --macro_duration 200 --num_rollouts 400 --warmup_time 1 --train_time 30 --replay False DashMetaAgent

