import torch
import torch.nn as nn


class CAModel(nn.Module):
    def __init__(self, n_channels=16, hidden_channels=128, fire_rate=0.5, device=None):
        super().__init__()

        self.fire_rate = 0.5
        self.n_channels = n_channels
        self.device = device or torch.device("cpu")

        # Perceive step
        sobel_filter_ = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
        scalar = 8.0

        sobel_filter_x = sobel_filter_ / scalar
        sobel_filter_y = sobel_filter_.t() / scalar
        identity_filter = torch.tensor([[0, 0, 0],[0, 1, 0],[0, 0, 0],], dtype=torch.float32,)
        filters = torch.stack([identity_filter, sobel_filter_x, sobel_filter_y])  # (3, 3, 3)
        filters = filters.repeat((n_channels, 1, 1))  # (3 * n_channels, 3, 3)
        self.filters = filters[:, None, ...].to(self.device)  # (3 * n_channels, 1, 3, 3)

        # Update step
        self.update_module = nn.Sequential(
                nn.Conv2d(3 * n_channels, hidden_channels, kernel_size=1),
                nn.ReLU(),
                nn.Conv2d(hidden_channels, n_channels, kernel_size=1, bias=False))

        with torch.no_grad():
            self.update_module[2].weight.zero_()

        self.to(self.device)

    def perceive(self, x):
        return nn.functional.conv2d(x, self.filters, padding=1, groups=self.n_channels)

    def update(self, x):
        return self.update_module(x)

    @staticmethod
    def stochastic_update(x, fire_rate):
        device = x.device
        mask = (torch.rand(x[:, :1, :, :].shape) <= fire_rate).to(device, torch.float32)
        return x * mask  # broadcasted over all channels

    @staticmethod
    def get_living_mask(x):
        return (nn.functional.max_pool2d(x[:, 0:1, :, :], kernel_size=3, stride=1, padding=1) > 0.1)

    def forward(self, x):
        # Rozdělení tensoru na první tři kanály a zbytek
        x_fixed = x[:, :3, :, :]  # První tři kanály, které nechceme měnit
        x_to_modify = x[:, 3:, :, :]  # Zbytek kanálů pro modifikace

        pre_life_mask = self.get_living_mask(x_to_modify)

        y = self.perceive(x)
        dx = self.update(y)
        dx = self.stochastic_update(dx, fire_rate=self.fire_rate)

        x_to_modify = x_to_modify + dx[:,3:,:,:]

        post_life_mask = self.get_living_mask(x_to_modify)
        life_mask = (pre_life_mask & post_life_mask).to(torch.float32)

        # Aplikace masky života a opětovné spojení s původními kanály
        x_modified = x_to_modify * life_mask
        x = torch.cat([x_fixed, x_modified], dim=1)

        return x
