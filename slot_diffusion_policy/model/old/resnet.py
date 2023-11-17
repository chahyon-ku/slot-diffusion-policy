import torch
import torchvision


def resnet18():
    model = torchvision.models.resnet18(pretrained=True)
    
    def _forward_impl(self, x: torch.Tensor) -> torch.Tensor:
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        # remove spatial pooling
        # x = self.avgpool(x)
        # x = torch.flatten(x, 1)
        # x = self.fc(x)

        return x
    
    model.forward = _forward_impl.__get__(model)
    return model