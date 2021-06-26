set -e
modality=$1
run_idx=$2
gpu=$3

for i in `seq 1 1 1`;
do
# python train_baseline.py --dataset_mode=multimodal
cmd="python train_baseline_nocv.py --dataset_mode=MovieData_v7_c4_multimodal --model=attention_fusion
--gpu_ids=$gpu --modality=$modality
--log_dir=./logs --checkpoints_dir=./checkpoints --print_freq=100 --save_latest_freq=1000
--A_type=comparE --cnn_a=EncCNN1d --enc_channel_a=128 --input_dim_a=130 --norm_method=trn --embd_size_a=128 --embd_method_a=maxpool
--V_type=denseface --input_dim_v=342 --embd_size_v=128  --embd_method_v=maxpool
--L_type=bert --input_dim_l=1024 --embd_size_l=128
--output_dim=4 --cls_layers=128,128 --dropout_rate=0.3
--niter=20 --niter_decay=20 --verbose --beta1=0.9 --init_type normal
--batch_size=256 --lr=5e-4 --run_idx=$run_idx
--name=MovieData_v7_c4_attention_5e-4 --suffix={modality}_run{run_idx}
--cvNo=$i"


echo "\n-------------------------------------------------------------------------------------"
echo "Execute command: $cmd"
echo "-------------------------------------------------------------------------------------\n"
echo $cmd | sh

done

#set -e：这句语句告诉bash如果任何语句的执行结果不是true则应该退出。

#dataset_mode是'x'：数据集采用data/[x]_dataset.py文件中的类[x（除掉下划线）]Dataset
#model是'x'：模型采用models/[x]_model.py文件中的类[x（除掉下划线）]Model
#name是实验名称
#--has_test：有测试过程

#运行命令：bash scripts/attn_fusion_MovieData_v7_c4.sh AVL 1 0。表示跑AVL三模态融合的实验, 实验编号是1, 在0号卡上跑