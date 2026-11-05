"""Source cells 91, 120; device and loss are explicit arguments."""
import torch

def evaluate(model, loader, criterion, device):
    model.eval()

    total_loss = 0.0
    total_correct = 0
    total_count = 0

    with torch.no_grad():
        for reviews, lengths, labels in loader:
            reviews = reviews.to(device)
            labels = labels.to(device)

            logits = model(reviews, lengths)
            loss = criterion(logits, labels)

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

def collect_predictions(model, loader, device):
    model.eval()

    predictions = []
    positive_probs = []

    with torch.no_grad():
        for reviews, lengths, labels in loader:
            logits = model(reviews.to(device), lengths)
            probs = torch.softmax(logits, dim=1)

            predictions.extend(
                logits.argmax(dim=1).cpu().tolist()
            )
            positive_probs.extend(
                probs[:, 1].cpu().tolist()
            )

    return predictions, positive_probs
