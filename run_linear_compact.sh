d_index=$1 # 2^1 to 2^10
d_theta=10 # 10 50
n_expe=200 # {200, 1000}
# scheme="ts"

python scripts/run_linear_compact.py --n-expe ${n_expe} --game CompactLinear \
       	--time-period 1000 --d-index ${d_index} \
       	--d-theta ${d_theta} --scheme ts

python scripts/run_linear_compact.py --n-expe ${n_expe} --game CompactLinear \
       	--time-period 1000 --d-index ${d_index} \
       	--d-theta ${d_theta} --scheme ots
# Zhang, FreqRusso, Russo, movieLens, Synthetic-v1, Synthetic-v2
# ps -ef | grep DeepSea | awk '{print $2}'| xargs kill -9
