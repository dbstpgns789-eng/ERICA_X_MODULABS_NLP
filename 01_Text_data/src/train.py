"""Source cell 90; optimizer, loss and device are explicit arguments."""
import torch

def train_one_epoch(model, loader, optimizer, criterion, device):
    model.train()

    total_loss = 0.0
    total_correct = 0
    total_count = 0

    for reviews, lengths, labels in loader:
        reviews = reviews.to(device)
        labels = labels.to(device)

        # 이전 배치에서 계산한 기울기를 비웁니다.
        optimizer.zero_grad()

        logits = model(reviews, lengths)
        loss = criterion(logits, labels)

        # 현재 배치의 오차로 기울기를 계산하고 모델을 수정합니다.
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        batch_size = labels.size(0)
        total_loss += loss.item() * batch_size
        total_correct += (
            logits.argmax(dim=1) == labels
        ).sum().item()
        total_count += batch_size

    return (
        total_loss / total_count,
        total_correct / total_count
    )
