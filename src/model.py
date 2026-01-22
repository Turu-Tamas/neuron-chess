import torch.nn as nn

class ChessEncoder(nn.Module):
    def __init__(self, embedding_dim=128):
        super(ChessEncoder, self).__init__()
        
        # Conv block
        self.conv_block = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(128),
            nn.Conv2d(128, 64, kernel_size=3, padding=1),
            nn.ReLU()
        )

        # MLP
        self.mlp = nn.Sequential(
            nn.Flatten(),  # 64 * 8 * 8 = 4096
            nn.Linear(4096, 512),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(512, embedding_dim) 
        )

    def forward(self, x):
        x = self.conv_block(x)
        x = self.mlp(x)
        return x