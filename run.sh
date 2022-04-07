export CUDA_VISIBLE_DEVICES=$1
game=$2

lr=0.001
noise_dim=2
batch_size=32
update_num=100
repeat_num=500
time_period=200
n_expe=100
n_context=0
optim=Adam

tag=$(date "+%Y%m%d%H%M%S")
python main.py --game ${game} \
    --lr=${lr} --noise-dim=${noise_dim} --batch-size=${batch_size} \
    --update-num=${update_num} --repeat-num=${repeat_num} --n-context=${n_context} \
    --time-period=${time_period} --n-expe=${n_expe} --optim=${optim} \
    > ~/logs/${game}_${tag}_3.out 2> ~/logs/${game}_${tag}_3.err &
echo "run $game $tag"

# Zhang, FreqRusso, Russo, movieLens
# ps -ef | grep DeepSea | awk '{print $2}'| xargs kill -9
