import torch
import swanlab


# 在测试集上评估
def test(model, test_loader, device, combined_loss, visualize_predictions):
    model.eval()
    test_loss = 0
    test_acc = 0

    with torch.no_grad():
        for images, masks in test_loader:
            images, masks = images.to(device), masks.to(device)
            outputs = model(images)
            loss = combined_loss(outputs, masks)
            test_loss += loss.item()
            test_acc += (outputs.round() == masks).float().mean().item()

    test_loss /= len(test_loader)
    test_acc /= len(test_loader)
    print(f"Test Loss: {test_loss:.4f}, Test Accuracy: {test_acc:.4f}")
    swanlab.log({"test/loss": test_loss, "test/acc": test_acc})

    # 可视化预测结果
    visualize_predictions(model, test_loader, device, num_samples=10)
