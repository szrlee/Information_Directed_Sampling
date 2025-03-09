#!/bin/bash

# Parameters
# d_theta=100 # 10 50 100
n_arms=10000  # 100 1000 10000
n_expe=1000  # {50, 200, 1000}
time_period=2000
game='Russo' # {Zhang, FreqRusso, Russo, movieLens, ChangingRusso}
cuda_id=$1

for d_theta in 90 100; do
    echo "d_theta: $d_theta"
    
    # First loop for M in {0 1,2,4,8,10}
    for M in 0 1 2 4 6 8; do
        d_index=$M  # No spaces around = in bash assignment
        
        echo "d_index: $d_index"
        
        CUDA_VISIBLE_DEVICES=${cuda_id} python scripts/run_linear_scaling.py \
            --n-expe ${n_expe} \
            --game ${game} \
            --time-period ${time_period} --d-index ${d_index} \
            --d-theta ${d_theta} --n-arms ${n_arms}
    done
    
    # Second loop from 20 to 100 with step 10
    for M in $(seq 10 2 30); do
        d_index=$M  # No spaces around = in bash assignment
        
        echo "d_index: $d_index"
        
        CUDA_VISIBLE_DEVICES=${cuda_id} python scripts/run_linear_scaling.py \
            --n-expe ${n_expe} \
            --game ${game} \
            --time-period ${time_period} --d-index ${d_index} \
            --d-theta ${d_theta} --n-arms ${n_arms}
    done
done
