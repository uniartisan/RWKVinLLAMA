export MASTER_ADDR=192.168.1.35
export MASTER_PORT=29500
export NUM_NODES=2
export WORLD_SIZE=7

# InfiniBand 相关设置
export NCCL_IB_DISABLE=0
export NCCL_IB_HCA=mlx5_0,mlx5_1
export NCCL_IB_GID_INDEX=3
export NCCL_SOCKET_IFNAME=ibp2s0f0,ibp2s0f1

# DeepSpeed 和 PyTorch 相关设置
export CUDA_DEVICE_MAX_CONNECTIONS=1

# 节点特定设置
export NODE_RANK=0
export NUM_GPUS=4
