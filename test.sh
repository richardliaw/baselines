#!/bin/bash

mpirun -np 8 /home/ubuntu/anaconda3/bin/python -m baselines.ppo1.run_mujoco_1_8 2>&1 | tee -a log_8.txt
mpirun -np 16 /home/ubuntu/anaconda3/bin/python -m baselines.ppo1.run_mujoco_2_16 2>&1 | tee -a log_16.txt
mpirun -np 32 /home/ubuntu/anaconda3/bin/python -m baselines.ppo1.run_mujoco_4_32 2>&1 | tee -a log_32.txt
