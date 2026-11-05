"""Review tensors and dynamic padding. Source cells 84, 86."""
import torch
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence

class ReviewDataset(Dataset):
    def __init__(self, sequences, labels):
        self.sequences = sequences
        self.labels = labels

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        review = torch.tensor(
            self.sequences[idx], dtype=torch.long
        )
        label = torch.tensor(
            self.labels[idx], dtype=torch.long
        )
        return review, label

def collate_batch(batch, pad_id=1):
    # 리뷰·정답 쌍들을 각각 모읍니다.
    reviews, labels = zip(*batch)

    # PAD를 넣기 전의 실제 길이를 보관합니다.
    lengths = torch.tensor(
        [len(review) for review in reviews],
        dtype=torch.long
    )

    # 이번 배치의 최장 리뷰에 맞춰 PAD를 채웁니다.
    padded_reviews = pad_sequence(
        reviews,
        batch_first=True,
        padding_value=pad_id
    )

    labels = torch.stack(labels)

    return padded_reviews, lengths, labels
