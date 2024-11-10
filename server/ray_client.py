# client.py
import ray
import torch
from typing import List, Optional
import logging
import time
from functools import wraps
import os
os.environ["NCCL_DEBUG"] = "INFO"
os.environ["NCCL_SOCKET_IFNAME"] = "eno1"  # 根据你的网络接口名称进行调整
os.environ["NCCL_IB_DISABLE"] = "0"        # 如果没有InfiniBand网络，可以禁用IB
# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('InferenceClient')

logger.info(f"InfiniBand interfaces: {os.listdir('/sys/class/infiniband/')}")
logger.info(f"NCCL version: {torch.cuda.nccl.version()}")
def retry_on_failure(max_retries=3, delay=1):
    """重试装饰器"""
    def decorator(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            last_exception = None
            for attempt in range(max_retries):
                try:
                    return await func(*args, **kwargs)
                except Exception as e:
                    last_exception = e
                    logger.warning(f"Attempt {attempt + 1} failed: {str(e)}")
                    if attempt < max_retries - 1:
                        time.sleep(delay)
                    continue
            raise last_exception
        return wrapper
    return decorator

class InferenceClient:
    def __init__(self, ray_address: str):
        """初始化推理客户端
        
        Args:
            ray_address: Ray集群地址
            num_gpus: 服务器上使用的GPU数量
        """
        self.ray_address = ray_address
        self.actor = None
        self._connect()
    
    def _connect(self):
        """连接到Ray集群"""
        try:
            # 初始化Ray客户端
            from ray.runtime_env import RuntimeEnv
            env = RuntimeEnv(env_vars={
                "NCCL_DEBUG": "INFO",
                "NCCL_SOCKET_IFNAME": "eno1",
                "NCCL_IB_DISABLE": "0",
                "NCCL_NET_GDR_LEVEL": "5",
                "NCCL_P2P_LEVEL": "NVL",
                "NCCL_SOCKET_FAMILY": "AF_INET",
                "NCCL_IB_CUDA_SUPPORT": "1",
                "NCCL_IB_GID_INDEX": "3",
                "NCCL_IB_HCA": "mlx5_0,mlx5_1"
            }
        )
            ray.init(address=self.ray_address, runtime_env=env)
            logger.info(f"Connected to Ray cluster at {self.ray_address}")
            
            # 获取已存在的TeacherActor引用
            actor_name = "myteacher"
            self.actor = ray.get_actor(actor_name,namespace="inference_server")
            logger.info(f"Found {self.actor} existing TeacherActors")
            
        except Exception as e:
            logger.error(f"Failed to connect to Ray cluster: {str(e)}")
            raise

    def _check_connection(self):
        """检查连接状态"""
        if not self.actor:
            raise ConnectionError("Not connected to Ray cluster")
        return True

    @retry_on_failure(max_retries=3)
    async def infer(self, input_ids: torch.Tensor) -> torch.Tensor:
        """异步推理接口
        
        Args:
            input_ids: 输入tensor
            
        Returns:
            logits: 输出logits tensor
        """
        self._check_connection()
        
        try:
            # 使用ray.get异步获取结果
            start = time.time()
            future = self.actor.forward.remote(input_ids)
            result = ray.get(future)
            end = time.time()
            logger.info(f"Inference time elapsed: {end-start}")
            return result
            
        except Exception as e:
            logger.error(f"Inference failed: {str(e)}")
            raise

    def close(self):
        """关闭客户端连接"""
        if ray.is_initialized():
            ray.shutdown()
            logger.info("Client connection closed")


if __name__ == "__main__":
    import asyncio
    
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--ray-address", type=str, required=True, help="Ray cluster address")
    
    args = parser.parse_args()
    
    # 创建客户端实例
    client = InferenceClient(
        ray_address=args.ray_address,
    )
    
    try:
        # 运行测试推理
        input_ids = torch.randint(0, 1000, (1, 512))
        result = asyncio.run(client.infer(input_ids))
        print(f"Test inference successful, output shape: {result.shape}")
        
    except Exception as e:
        print(f"Test inference failed: {str(e)}")
        
    finally:
        client.close()
