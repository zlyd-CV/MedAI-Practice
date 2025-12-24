import swanlab
import torch


def setup_swanlab():
    # 创建一个SwanLab项目
    swanlab.login(api_key="请填写你的API Key")

    swanlab.init(
        # 设置项目名
        project="Unet-Medical-Segmentation",
        # 设置实验描述
        experiment_name="bs32-epoch40",
        # 设置超参数
        config={
            "batch_size": 32,
            "learning_rate": 1e-4,
            "num_epochs": 40,
            "device": "cuda" if torch.cuda.is_available() else "cpu",
        }
    )
