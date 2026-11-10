
import torch
import torch.nn as nn


class MusicLSTM(nn.Module):

    def __init__(
        self,
        vocab_size=128,
        embedding_dim=64,
        hidden_size=256,
        num_layers=2,
        dropout=0.2
    ):
        super(MusicLSTM, self).__init__()

        # Convert note numbers into embeddings
        self.embedding = nn.Embedding(
            vocab_size,
            embedding_dim
        )

        # LSTM architecture
        self.lstm = nn.LSTM(
            input_size=embedding_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout
        )

        # Predict the next musical note
        self.fc = nn.Linear(
            hidden_size,
            vocab_size
        )

    def forward(self, x):

        x = self.embedding(x)

        output, (hidden, cell) = self.lstm(x)

        # Use the final time-step output
        output = output[:, -1, :]

        output = self.fc(output)

        return output


# Test model independently
if __name__ == "__main__":

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    model = MusicLSTM().to(device)

    sample = torch.randint(
        0, 128, (4, 32)
    ).to(device)

    prediction = model(sample)

    print("Device:", device)
    print("Model:", model)
    print("Input shape:", sample.shape)
    print("Output shape:", prediction.shape)
    print("LSTM model created successfully!")
