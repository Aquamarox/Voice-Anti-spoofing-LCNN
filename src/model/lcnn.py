import torch
from torch import nn


class MFM(nn.Module):
    """
    Max-Feature-Map activation.

    Split features into two equal parts and take
    the element-wise maximum between them.
    """

    def __init__(self, dim=1):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        if x.shape[self.dim] % 2 != 0:
            raise ValueError(
                "MFM input size must be divisible by 2, "
                f"got {x.shape[self.dim]}"
            )

        first_half, second_half = torch.chunk(
            x,
            chunks=2,
            dim=self.dim,
        )

        return torch.maximum(first_half, second_half)


class LCNN(nn.Module):
    """
    Light CNN for binary audio anti-spoofing classification.

    Expected input shape:
        [batch_size, 1, 257, 750]
    """

    def __init__(
        self,
        n_class=2,
        input_height=257,
        input_width=750,
        dropout=0.75,
    ):
        super().__init__()

        self.input_height = input_height
        self.input_width = input_width

        self.features = nn.Sequential(
            nn.Conv2d(
                in_channels=1,
                out_channels=64,
                kernel_size=5,
                padding=2,
            ),
            MFM(),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(
                in_channels=32,
                out_channels=64,
                kernel_size=1,
            ),
            MFM(),
            nn.BatchNorm2d(32),

            nn.Conv2d(
                in_channels=32,
                out_channels=96,
                kernel_size=3,
                padding=1,
            ),
            MFM(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.BatchNorm2d(48),

            nn.Conv2d(
                in_channels=48,
                out_channels=96,
                kernel_size=1,
            ),
            MFM(),
            nn.BatchNorm2d(48),

            nn.Conv2d(
                in_channels=48,
                out_channels=128,
                kernel_size=3,
                padding=1,
            ),
            MFM(),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(
                in_channels=64,
                out_channels=128,
                kernel_size=1,
            ),
            MFM(),
            nn.BatchNorm2d(64),

            nn.Conv2d(
                in_channels=64,
                out_channels=64,
                kernel_size=3,
                padding=1,
            ),
            MFM(),
            nn.BatchNorm2d(32),

            nn.Conv2d(
                in_channels=32,
                out_channels=64,
                kernel_size=1,
            ),
            MFM(),
            nn.BatchNorm2d(32),

            nn.Conv2d(
                in_channels=32,
                out_channels=64,
                kernel_size=3,
                padding=1,
            ),
            MFM(),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

        feature_height = input_height // 16
        feature_width = input_width // 16
        flattened_size = 32 * feature_height * feature_width

        self.classifier = nn.Sequential(
            nn.Flatten(),

            nn.Linear(flattened_size, 160),
            MFM(),

            # Dropout before final BatchNorm
            nn.Dropout(p=dropout),
            nn.BatchNorm1d(80),

            nn.Linear(80, n_class),
        )

        self.apply(self._initialize_weights)

    @staticmethod
    def _initialize_weights(module):
        if isinstance(module, (nn.Conv2d, nn.Linear)):
            nn.init.kaiming_normal_(module.weight)

            if module.bias is not None:
                nn.init.zeros_(module.bias)

        elif isinstance(module, (nn.BatchNorm1d, nn.BatchNorm2d)):
            nn.init.ones_(module.weight)
            nn.init.zeros_(module.bias)

    def forward(self, data_object, **batch):
        expected_shape = (
            self.input_height,
            self.input_width,
        )

        if data_object.ndim != 4:
            raise ValueError(
                "LCNN expects a 4D tensor "
                "[batch, channel, frequency, time], "
                f"got shape {tuple(data_object.shape)}"
            )

        if data_object.shape[1] != 1:
            raise ValueError(
                "LCNN expects one input channel, "
                f"got {data_object.shape[1]}"
            )

        if data_object.shape[-2:] != expected_shape:
            raise ValueError(
                f"LCNN expects spatial shape {expected_shape}, "
                f"got {tuple(data_object.shape[-2:])}"
            )

        feature_maps = self.features(data_object)
        logits = self.classifier(feature_maps)

        return {"logits": logits}

    def __str__(self):
        all_parameters = sum(
            parameter.numel()
            for parameter in self.parameters()
        )

        trainable_parameters = sum(
            parameter.numel()
            for parameter in self.parameters()
            if parameter.requires_grad
        )

        result = super().__str__()
        result += f"\nAll parameters: {all_parameters}"
        result += f"\nTrainable parameters: {trainable_parameters}"

        return result