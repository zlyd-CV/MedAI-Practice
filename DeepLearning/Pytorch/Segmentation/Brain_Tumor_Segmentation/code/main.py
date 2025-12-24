import swanlab
from swanlab_setting import setup_swanlab
from torchvision.transforms import transforms
from train import train_model, combined_loss
from network import UNet
from read_dataset import load_coco_datasets
from test import test
import torch
from utils import visualize_predictions


def main():
    # 执行 SwanLab 的相关设置
    setup_swanlab()
    # 设置设备
    device = torch.device(swanlab.config["device"])

    # 数据预处理
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize((256, 256)),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # 创建数据集
    train_loader, val_loader, test_loader = load_coco_datasets(transform=transform)

    batch_size = swanlab.config["batch_size"]
    model = UNet(n_filters=32).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=swanlab.config["learning_rate"])

    # 训练模型
    train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=combined_loss,
        optimizer=optimizer,
        num_epochs=swanlab.config["num_epochs"],
        device=device,
    )

    # 在测试集上评估
    test(model, test_loader, device, combined_loss, visualize_predictions)


if __name__ == "__main__":
    main()
