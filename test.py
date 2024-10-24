# This is a test python file
# Can try any code block under "if __name__ == "__main__":"
import torch
import torch.nn as nn


if __name__ == "__main__":
    # Example of target with class indices
    loss0 = nn.CrossEntropyLoss(reduction='none')
    loss1 = nn.CrossEntropyLoss(reduction='mean')
    loss2 = nn.CrossEntropyLoss(reduction='sum')
    input = torch.randn(3, 3, requires_grad=True)
    target = torch.empty(3, dtype=torch.long).random_(3)
    output0 = loss0(input, target)
    output1 = loss1(input, target)
    output2 = loss2(input, target)
    # Example of target with class probabilities
    loss0 = nn.CrossEntropyLoss(reduction='none')
    loss1 = nn.CrossEntropyLoss(reduction='mean')
    loss2 = nn.CrossEntropyLoss(reduction='sum')
    input = torch.randn(3, 5, requires_grad=True)
    target = torch.randn(3, 5).softmax(dim=1)
    output0 = loss0(input, target)
    output1 = loss1(input, target)
    output2 = loss2(input, target)
    output0.backward()
