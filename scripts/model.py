import torch
import torch.nn as nn

class TinyPolicy(nn.Module):
    """
    Behavioral Cloning Policy for Egocentric Pick-and-Place Manipulation.
    Input: [hand_x, hand_y, hand_z, cube_x, cube_y, bowl_x, bowl_y] (7-dim state)
    Output: [d_x, d_y, d_z, d_gripper] (4-dim action)
    """
    def __init__(self, input_dim=7, hidden_dim=128, output_dim=4):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, output_dim)
        )

    def forward(self, x):
        return self.net(x)
