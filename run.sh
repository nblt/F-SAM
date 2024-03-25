device=0
seed=1
datasets=CIFAR100
model=resnet18 # resnet18 VGG16BN WideResNet28x10
schedule=cosine
wd=0.001
epoch=200
bz=128
rho=0.2
alpha=-1
eta=0.95
opt=FriendlySAM # FriendlySAM SAM
DST=results_final_final/$opt/$datasets/$model/$opt\_cutout\_$rho\_$alpha\_$eta\_$epoch\_$model\_bz$bz\_wd$wd\_$datasets\_$schedule\_seed$seed

CUDA_VISIBLE_DEVICES=$device python -u train_sam_cos_step.py --datasets $datasets \
        --arch=$model --epochs=$epoch --wd=$wd --randomseed $seed --lr 0.05 --rho $rho --optimizer $opt \
        --save-dir=$DST/checkpoints --log-dir=$DST -p 200 --schedule $schedule -b $bz \
        --cutout --alpha $alpha --eta $eta