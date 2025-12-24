import torch
import torch.nn as nn
import swanlab



# 固定加权Dice 损失和二元交叉熵损失（BCE）
def dice_loss(pred, target, smooth=1e-6):
    pred_flat = pred.view(-1)
    target_flat = target.view(-1)
    intersection = (pred_flat * target_flat).sum()
    return 1 - ((2. * intersection + smooth) / (pred_flat.sum() + target_flat.sum() + smooth))


def combined_loss(pred, target):
    dice = dice_loss(pred, target)
    bce = nn.BCELoss()(pred, target)
    return 0.6 * dice + 0.4 * bce


# 训练函数
def train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs, device):
    best_val_loss = float('inf')  # 初始化最佳验证损失
    patience = 8  # 设置早停的耐心值
    patience_counter = 0  # 计数器初始化

    for epoch in range(num_epochs):
        model.train()
        train_loss = 0
        train_acc = 0

        for images, masks in train_loader:
            images, masks = images.to(device), masks.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, masks)

            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            train_acc += (outputs.round() == masks).float().mean().item()  # round四舍五入预测结果,这里计算预测正确的像素个数

        train_loss /= len(train_loader) # 计算平均损失
        train_acc /= len(train_loader) # 计算平均准确率

        # 验证
        model.eval()
        val_loss = 0
        val_acc = 0

        with torch.no_grad():
            for images, masks in val_loader:
                images, masks = images.to(device), masks.to(device)
                outputs = model(images)
                loss = criterion(outputs, masks)

                val_loss += loss.item()
                val_acc += (outputs.round() == masks).float().mean().item()

        val_loss /= len(val_loader)
        val_acc /= len(val_loader)

        # 记录指标到SwanLab
        swanlab.log(
            {
                "train/loss": train_loss,
                "train/acc": train_acc,
                "train/epoch": epoch + 1,  # 从1开始计数,符合逻辑
                "val/loss": val_loss,
                "val/acc": val_acc,
            },
            step=epoch + 1)

        print(f'Epoch {epoch + 1}/{num_epochs}:')
        print(f'Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}')
        print(f'Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}')

        # 早停
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0  # 重置计数器
            torch.save(model.state_dict(), 'best_model.pth')
        else:
            patience_counter += 1  # 增加计数器
            if patience_counter >= patience:  # 达到耐心值，停止训练
                print("Early stopping triggered")
                break
