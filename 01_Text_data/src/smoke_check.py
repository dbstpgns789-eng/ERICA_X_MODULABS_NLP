"""Small CPU integration check, not a repetition of the NSMC experiments."""
import json
import torch
from torch import nn
from torch.utils.data import DataLoader
from .dataset import ReviewDataset, collate_batch
from .model import SentimentLSTM
from .train import train_one_epoch
from .evaluate import evaluate, collect_predictions


def main():
    torch.manual_seed(42)
    dataset = ReviewDataset([[2, 3, 4], [5], [6, 7], [8, 9, 10, 11]], [0, 1, 0, 1])
    loader = DataLoader(dataset, batch_size=2, shuffle=False, collate_fn=collate_batch)
    reviews, lengths, labels = next(iter(loader))
    assert reviews.shape == (2, 3)
    assert lengths.tolist() == [3, 1]
    assert reviews[1].tolist() == [5, 1, 1]
    device = torch.device("cpu")
    model = SentimentLSTM(16, 1).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    before = model.classifier.weight.detach().clone()
    train_loss, train_acc = train_one_epoch(model, loader, optimizer, criterion, device)
    assert not torch.equal(before, model.classifier.weight.detach())
    assert torch.count_nonzero(model.embedding.weight[1]).item() == 0
    val_loss, val_acc = evaluate(model, loader, criterion, device)
    predictions, probabilities = collect_predictions(model, loader, device)
    assert len(predictions) == len(probabilities) == 4
    assert all(0 <= p <= 1 for p in probabilities)
    assert all(torch.isfinite(torch.tensor(v)) for v in [train_loss, val_loss])
    print(json.dumps({"smoke_check": "passed", "torch": torch.__version__,
                      "device": "cpu", "samples": 4,
                      "checks": ["dynamic padding", "forward/backward/update",
                                 "PAD embedding", "evaluation", "prediction order/count"]}))


if __name__ == "__main__":
    main()
