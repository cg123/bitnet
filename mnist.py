from typing import Tuple

import safetensors.torch
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data
import torchvision
import tqdm

import wandb
from bitlinear import BitConv2d, BitLinear, init_bitnet


class BitEuler(nn.Module):
    def __init__(self, features: int, intermediate: int, max_iter: int = 10):
        super().__init__()
        self.max_iter = max_iter
        self.net = nn.Sequential(
            BitLinear(features, intermediate),
            nn.SiLU(),
            BitLinear(intermediate, features),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for _ in range(self.max_iter):
            dx = self.net(x)
            x = x + dx / self.max_iter

            if not self.training and torch.allclose(dx, torch.zeros_like(dx)):
                break

        return x


class BitMNIST(nn.Module):
    def __init__(
        self,
        num_classes: int = 10,
        eps: float = 1e-7,
    ):
        super().__init__()
        self.conv1 = BitConv2d(1, 32, eps=eps, kernel_size=5, stride=1, padding=1)
        self.conv2 = BitConv2d(32, 64, eps=eps, kernel_size=5, stride=1, padding=1)
        self.fc = nn.Sequential(
            BitLinear(64 * 5 * 5, 64, eps=eps),
            nn.SiLU(),
            # BitLinear(256, 128, eps=eps),
            # nn.ReLU(),
            BitEuler(64, 256, max_iter=16),
            BitLinear(64, num_classes, eps=eps),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2, 2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2, 2)
        x = x.view(x.shape[0], -1)
        x = self.fc(x)
        return F.log_softmax(x.float(), dim=1)


def _dataloaders(batch_size: int = 2048):
    transform = torchvision.transforms.Compose(
        [
            torchvision.transforms.ToTensor(),
            # mnist
            # torchvision.transforms.Normalize((0.1307,), (0.3081,)),
            # emnist balanced
            torchvision.transforms.Normalize((0.1751,), (0.0128,)),
        ]
    )
    trainset = torchvision.datasets.EMNIST(
        root="./data", split="byclass", train=True, download=True, transform=transform
    )
    testset = torchvision.datasets.EMNIST(
        root="./data", split="byclass", train=False, download=True, transform=transform
    )
    trainloader = torch.utils.data.DataLoader(
        trainset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
        pin_memory_device="cuda",
    )
    testloader = torch.utils.data.DataLoader(
        testset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
        pin_memory_device="cuda",
    )
    return trainloader, testloader


class RandomColorCurve(nn.Module):
    def __init__(self, power_range: Tuple[float, float] = (0.5, 2.0)):
        super().__init__()
        self.power_range = power_range

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        power = (
            torch.empty(x.shape[0], 1, 1, 1).uniform_(*self.power_range).to(x.device)
        )
        return (x.abs() ** power) * x.sign()


def _train(model, trainloader, optimizer, current_step: int) -> int:
    bilinear = torchvision.transforms.InterpolationMode.BILINEAR
    honker = torchvision.transforms.Compose(
        [
            torchvision.transforms.RandomApply(
                [
                    torchvision.transforms.RandomRotation(10, interpolation=bilinear),
                    torchvision.transforms.RandomAffine(
                        0,
                        translate=(0.1, 0.1),
                        scale=(0.9, 1.01),
                        interpolation=bilinear,
                    ),
                ],
                p=0.75,
            ),
            RandomColorCurve(),
        ]
    )
    model.train()
    for batch_idx, (data, target) in tqdm.tqdm(
        enumerate(trainloader), total=len(trainloader), desc="Training"
    ):
        optimizer.zero_grad()
        data = honker(data.to(model.conv1.weight.device)).to(model.conv1.weight.dtype)
        output = model(data)
        loss = F.nll_loss(output, target.to(output.device))
        loss.backward()
        optimizer.step()

        wandb.log({"train_loss": loss.item()}, step=current_step + batch_idx)
        if batch_idx % 10 == 0 or batch_idx == len(trainloader) - 1:
            wandb.log(
                {"learning_rate": optimizer.param_groups[0]["lr"]},
                step=current_step + batch_idx,
            )

    return current_step + batch_idx


def _test(model, testloader, current_step: int):
    model.eval()
    test_loss = 0
    correct = 0
    top3 = 0
    with torch.no_grad():
        for data, target in tqdm.tqdm(testloader, desc="Testing"):
            output = model(
                data.to(model.conv1.weight.device).to(model.conv1.weight.dtype)
            )
            test_loss += F.nll_loss(
                output.float(), target.to(output.device), reduction="sum"
            ).item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.to(output.device).view_as(pred)).sum().item()

            # compute top-3 accuracy
            pred = output.topk(3, dim=1).indices
            top3 += (
                pred.eq(target.to(output.device).view(-1, 1)).any(dim=1).sum().item()
            )

    test_loss /= len(testloader.dataset)
    accuracy = 100.0 * correct / len(testloader.dataset)
    top3_accuracy = 100.0 * top3 / len(testloader.dataset)
    wandb.log(
        {"test_loss": test_loss, "accuracy": accuracy, "accuracy_top3": top3_accuracy},
        step=current_step,
    )
    print(
        f"Test set: Average loss: {test_loss:.4f}, Accuracy: {correct}/{len(testloader.dataset)} ({accuracy:.2f}%), Top-3 Accuracy: {top3_accuracy:.2f}%"
    )
    return accuracy


def main():
    torch.manual_seed(467895)

    wandb.init(project="bitlinear-mnist")
    model = BitMNIST(num_classes=62).cuda()
    model.apply(init_bitnet)
    print(f"{sum(x.numel() for x in model.parameters()) // 1000}k parameters")

    model = torch.compile(model)

    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.95)
    trainloader, testloader = _dataloaders(batch_size=2048)

    current_step = 0
    for epoch in range(100):
        print(f"Epoch {epoch+1}")
        current_step = _train(model, trainloader, optimizer, current_step)
        wandb.log({"epoch": epoch}, step=current_step)
        _test(model, testloader, current_step)
        scheduler.step()

    # save the model
    safetensors.torch.save_model(model, "bitmnist.safetensors")
    wandb.finish()


if __name__ == "__main__":
    torch.set_float32_matmul_precision("high")
    main()
