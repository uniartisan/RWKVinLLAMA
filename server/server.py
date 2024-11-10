# server.py
import ray
import torch
import torch.distributed as dist
from transformers import AutoModelForCausalLM,AutoTokenizer
import os
import logging
import time
os.environ["NCCL_DEBUG"] = "INFO"
os.environ["NCCL_SOCKET_IFNAME"] = "eno1"  # 根据你的网络接口名称进行调整
os.environ["NCCL_IB_DISABLE"] = "0"        # 如果没有InfiniBand网络，可以禁用IB
# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('InferenceServer')

@ray.remote(num_gpus=1)
class TeacherActor:
    def __init__(self, model_name: str):
        """初始化Teacher Actor
        
        Args:
            model_name: Hugging Face模型名称
        """
        # 配置日志
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger('InferenceServer')
        self.model = AutoModelForCausalLM.from_pretrained(model_name,device_map="auto",attn_implementation="flash_attention_2", low_cpu_mem_usage=True, torch_dtype=torch.float16)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.logger.info(f"TeacherActor {self.model}/{self.tokenizer} initialization completed")


    
    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        """前向计算
        
        Args:
            input_ids: 输入tensor
            
        Returns:
            logits: 输出logits tensor
        """
        start = time.time()
        try:
            attention_mask = torch.ne(input_ids,self.tokenizer.pad_token_id)
            # 前向传播
            with torch.no_grad():
                outputs = self.model(input_ids=input_ids,attention_mask=attention_mask,use_cache=False,output_hidden_states=False)
                logits = outputs.logits
            end = time.time()
            self.logger.info(f"Time elapsed: {end-start}")
            return logits  # 返回CPU tensor便于传输
        except Exception as e:
            self.logger.error(f"Error in forward pass: {str(e)}")
            raise

    def health_check(self) -> bool:
        """健康检查"""
        return True

def init_server(
    model_name: str,
    num_gpus: int,
    ray_address: str = "auto"
):
    """初始化服务器
    
    Args:
        model_name: 要加载的模型名称
        num_gpus: 使用的GPU数量
        ray_address: Ray head node地址
    """
    try:
        # 初始化Ray
        """
        export NCCL_IB_HCA=mlx5_0,mlx5_1
        export NCCL_IB_GID_INDEX=3
        export NCCL_SOCKET_IFNAME=eno1
        export NCCL_DEBUG=INFO
        export NCCL_IB_CUDA_SUPPORT=1
        export NCCL_NET_GDR_LEVEL=5
        export NCCL_P2P_LEVEL=NVL
        export NCCL_SOCKET_FAMILY=AF_INET
        """
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
        ctx = ray.init(
                 namespace="inference_server",
                 include_dashboard=True,
                 dashboard_port=8265,
                 runtime_env=env,
                 _system_config={
                    "object_store_memory": 52428800,  # 50 GB
                    "max_direct_call_object_size": 100000000,
                    "min_direct_call_object_size": 1000,
                    "ray_backend": "nccl"
                })
        logger.info(f"Ray initialized at {ctx}")
        print(f"Using namespace: {ray.get_runtime_context().namespace}")
        
        # 创建TeacherActors
        actor = TeacherActor.options(num_gpus=num_gpus,name="myteacher").remote(model_name)

        logger.info(f"Created {actor} TeacherActors")
        
        # 保持程序运行
        import time
        while True:
            # 定期进行健康检查
            try:
                health_states = ray.get(actor.health_check.remote())
                logger.info("All actors healthy: " + str(health_states))
            except Exception as e:
                logger.error(f"Health check failed: {str(e)}")
            
            time.sleep(60)  # 每分钟检查一次

    except Exception as e:
        logger.error(f"Server initialization failed: {str(e)}")
        ray.shutdown()
        raise

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-name", type=str, required=True, help="Model name to load")
    parser.add_argument("--num-gpus", type=int, required=True, help="Number of GPUs to use")
    parser.add_argument("--ray-address", type=str, default="auto", help="Ray head node address")
    
    args = parser.parse_args()
    
    init_server(
        model_name=args.model_name,
        num_gpus=args.num_gpus,
        ray_address=args.ray_address
    )
    