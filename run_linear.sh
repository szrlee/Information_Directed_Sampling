# d_index=$1 # 2^1 to 2^10
d_theta=100 # 10 50
n_arms=10000 # 100 1000 10000
n_expe=200 # {200, 1000}
game='Russo' # {Zhang, FreqRusso, Russo, movieLens, ChangingRusso}
cuda_id=$1

# for loop d_index in 2^1 to 2^10, write bash script
for d_i in {1..10..2};
do
	d_index=$((2**${d_i}))

	echo "d_index: $d_index"

	CUDA_VISIBLE_DEVICES=${cuda_id} python scripts/run_linear.py \
			--n-expe ${n_expe} \
			--game ${game} \
			--time-period 1000 --d-index ${d_index} \
			--d-theta ${d_theta} --n-arms ${n_arms}
done