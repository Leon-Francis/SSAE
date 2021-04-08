import torch
from config import AttackConfig


class SoftArgmax1D(torch.nn.Module):
    """
    Implementation of a 1d soft arg-max function as an nn.Module, so that we can differentiate through arg-max operations.
    """
    def __init__(self, base_index=0, step_size=1):
        """
        The "arguments" are base_index, base_index+step_size, base_index+2*step_size, ... and so on for
        arguments at indices 0, 1, 2, ....
        Assumes that the input to this layer will be a batch of 1D tensors (so a 2D tensor).
        :param base_index: Remember a base index for 'indices' for the input
        :param step_size: Step size for 'indices' from the input
        """
        super(SoftArgmax1D, self).__init__()
        self.base_index = base_index
        self.step_size = step_size
        self.softmax = torch.nn.Softmax(dim=1)

    def forward(self, x):
        """
        Compute the forward pass of the 1D soft arg-max function as defined below:
        SoftArgMax(x) =  sum_i (i * softmax(x)_i)
        :param x: The input to the soft arg-max layer
        :return: Output of the soft arg-max layer
        """
        batch_size = x.shape[0]
        x = x.reshape(-1, x.shape[2])
        smax = self.softmax(x)
        end_index = self.base_index + x.size()[1] * self.step_size
        indices = torch.arange(start=self.base_index,
                               end=end_index,
                               step=self.step_size,
                               dtype=torch.float32).to(
                                   AttackConfig.train_device)
        result = torch.matmul(smax, indices)
        return torch.round(result.reshape(batch_size, -1)).long()


if __name__ == '__main__':
    soft_argmax = SoftArgmax1D()
    t = torch.rand(128, 10, 50000).to(AttackConfig.train_device)
    tt = soft_argmax(t)
    pass
