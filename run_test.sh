#!/bin/bash

for ((i=0;i<3;i+=1))
do
    caffeinate -i python3 main.py \
        --env_id="HumanoidStandup-v5" \
        --seed=$i \
        --reward_scale=20.0 \
        --entropy_coef=0.05 \
        --action_scale=0.4
done