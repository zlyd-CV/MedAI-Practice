import torch
import matplotlib.pyplot as plt
import random
import swanlab


def visualize_predictions(model, test_loader, device, num_samples=5, threshold=0.5):
    model.eval()
    with torch.no_grad():
        # 获取一个批次的数据
        images, masks = next(iter(test_loader))
        images, masks = images.to(device), masks.to(device)
        predictions = model(images)

        # 将预测结果转换为二值掩码
        binary_predictions = (predictions > threshold).float()

        # 选择前3个样本
        indices = random.sample(range(len(images)), min(num_samples, len(images)))
        indices = indices[:8]

        # 创建一个大图
        plt.figure(figsize=(15, 8))  # 调整图像大小以适应新增的行
        plt.suptitle(f'Epoch {swanlab.config["num_epochs"]} Predictions (Random 6 samples)')

        for i, idx in enumerate(indices):
            # 原始图像
            plt.subplot(4, 8, i * 4 + 1)  # 4行而不是3行
            img = images[idx].cpu().numpy().transpose(1, 2, 0)
            img = (img * [0.229, 0.224, 0.225] + [0.485, 0.456, 0.406]).clip(0, 1)
            plt.imshow(img)
            plt.title('Original Image')
            plt.axis('off')

            # 真实掩码
            plt.subplot(4, 8, i * 4 + 2)
            plt.imshow(masks[idx].cpu().squeeze(), cmap='gray')
            plt.title('True Mask')
            plt.axis('off')

            # 预测掩码
            plt.subplot(4, 8, i * 4 + 3)
            plt.imshow(binary_predictions[idx].cpu().squeeze(), cmap='gray')
            plt.title('Predicted Mask')
            plt.axis('off')

            # 新增：预测掩码叠加在原图上
            plt.subplot(4, 8, i * 4 + 4)
            plt.imshow(img)  # 先显示原图
            # 添加红色半透明掩码
            plt.imshow(binary_predictions[idx].cpu().squeeze(),
                       cmap='Reds', alpha=0.3)  # alpha控制透明度
            plt.title('Overlay')
            plt.axis('off')

        # 记录图像到SwanLab
        swanlab.log({"predictions": swanlab.Image(plt)})
