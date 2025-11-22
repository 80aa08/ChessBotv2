import torch
import torch.nn as nn
import torch.nn.functional as F
from config import Config

class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(channels)

    def forward(self, x):
        residual = x
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += residual
        out = F.relu(out)
        return out


class ChessNet(nn.Module):
    """
    Neural network for chess position evaluation
    Architecture inspired by AlphaZero

    Input: (batch, 17, 8, 8) board representation
    Output:
        - policy_logits: (batch, 4672) move probabilities
        - value: (batch, 1) position evaluation [-1, 1]
    """

    def __init__(self):
        super().__init__()

        self.conv_input = nn.Conv2d(
            Config.INPUT_CHANNELS,
            Config.CHANNELS,
            kernel_size=3,
            padding=1,
            bias=False
        )
        self.bn_input = nn.BatchNorm2d(Config.CHANNELS)

        self.residual_blocks = nn.ModuleList([
            ResidualBlock(Config.CHANNELS)
            for _ in range(Config.RESIDUAL_BLOCKS)
        ])

        self.policy_conv = nn.Conv2d(Config.CHANNELS, 32, kernel_size=1, bias=False)
        self.policy_bn = nn.BatchNorm2d(32)
        self.policy_fc = nn.Linear(32 * 8 * 8, Config.POLICY_OUTPUT_SIZE)

        self.value_conv = nn.Conv2d(Config.CHANNELS, 32, kernel_size=1, bias=False)
        self.value_bn = nn.BatchNorm2d(32)
        self.value_fc1 = nn.Linear(32 * 8 * 8, 256)
        self.value_fc2 = nn.Linear(256, 1)

        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = F.relu(self.bn_input(self.conv_input(x)))

        for block in self.residual_blocks:
            x = block(x)

        policy = F.relu(self.policy_bn(self.policy_conv(x)))
        policy = policy.view(policy.size(0), -1)  # Flatten
        policy_logits = self.policy_fc(policy)

        value = F.relu(self.value_bn(self.value_conv(x)))
        value = value.view(value.size(0), -1)  # Flatten
        value = F.relu(self.value_fc1(value))
        value = torch.tanh(self.value_fc2(value))

        return policy_logits, value

    def predict(self, state):
        self.eval()
        with torch.no_grad():
            if state.dim() == 3:
                state = state.unsqueeze(0)

            policy_logits, value = self.forward(state)
            policy = F.softmax(policy_logits, dim=1).squeeze(0)
            value = value.item()

        return policy.cpu().numpy(), value


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == '__main__':
    model = ChessNet()
    print(f"Model parameters: {count_parameters(model):,}")

    batch_size = 4
    x = torch.randn(batch_size, Config.INPUT_CHANNELS, 8, 8)
    policy_logits, value = model(x)

    print(f"Input shape: {x.shape}")
    print(f"Policy logits shape: {policy_logits.shape}")
    print(f"Value shape: {value.shape}")
    print(f"Value range: [{value.min().item():.3f}, {value.max().item():.3f}]")

    policy = F.softmax(policy_logits, dim=1)
    print(f"Policy sum: {policy.sum(dim=1)}")
