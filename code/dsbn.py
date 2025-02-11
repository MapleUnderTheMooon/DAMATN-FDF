from torch import nn


class _DomainSpecificBatchNorm(nn.Module):
    _version = 2
    # num_features 输出通道数
    def __init__(self, num_features, num_domains, eps=1e-5, momentum=0.1, affine=True,
                 track_running_stats=True):
        super(_DomainSpecificBatchNorm, self).__init__()
        self.bns = nn.ModuleList(
            [nn.BatchNorm3d(num_features, eps, momentum, affine, track_running_stats) for _ in range(num_domains)])
            # [nn.BatchNorm2d(num_features, eps, momentum, affine, track_running_stats) for _ in range(num_domains)])

    def reset_running_stats(self):
        for bn in self.bns:
            bn.reset_running_stats()

    def reset_parameters(self):
        for bn in self.bns:
            bn.reset_parameters()

    def _check_input_dim(self, input):
        raise NotImplementedError

    def forward(self, x, index):
        # index 1/0 表示有标签的数据或者是未标记的数据
        self._check_input_dim(x)
        # # 将五维的张量转化成四维的
        # x = x[:, 1, :, :, :]
        bn = self.bns[index]
        return bn(x), index


class DomainSpecificBatchNorm2d(_DomainSpecificBatchNorm):
    def _check_input_dim(self, input):
        # if input.dim() != 4:
        if input.dim() != 5:
#             raise ValueError('expected 4D input (got {}D input)'
            raise ValueError('expected 5D input (got {}D input)'
                             .format(input.dim()))