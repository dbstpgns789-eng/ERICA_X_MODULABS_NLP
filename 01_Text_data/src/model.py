"""The same Embedding-LSTM-Linear architecture as source cell 88."""
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence

class SentimentLSTM(nn.Module):
    def __init__(self, vocab_size, pad_id):
        super().__init__()

        self.embedding = nn.Embedding(
            num_embeddings=vocab_size,
            embedding_dim=128,
            padding_idx=pad_id
        )

        self.lstm = nn.LSTM(
            input_size=128,
            hidden_size=128,
            batch_first=True
        )

        self.classifier = nn.Linear(128, 2)

    def forward(self, reviews, lengths):
        embedded = self.embedding(reviews)

        # 실제 리뷰 길이를 알려 주어 LSTM이 뒤의 PAD를 읽지 않게 합니다.
        packed = pack_padded_sequence(
            embedded,
            lengths.cpu(),
            batch_first=True,
            enforce_sorted=False
        )

        _, (hidden, _) = self.lstm(packed)

        # 마지막 LSTM 층의 최종 은닉 상태로 분류합니다.
        logits = self.classifier(hidden[-1])
        return logits
