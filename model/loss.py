import torch
import torch.nn.functional as F
from torch.autograd import Variable
from torch.nn import MultiLabelSoftMarginLoss, BCEWithLogitsLoss
import numpy as np


def bce(y_input, y_target):
    loss = BCEWithLogitsLoss()
    return loss(y_input, y_target)


def multi_label(y_input, y_target):
    loss = MultiLabelSoftMarginLoss()
    return loss(y_input, y_target)


def focal(y_input, y_target):
    loss = FocalLoss()
    return loss(y_input, y_target)


class FocalLoss(torch.nn.Module):
    def __init__(self, gamma=2):
        super().__init__()
        self.gamma = gamma

    def forward(self, input, target):
        if not (target.size() == input.size()):
            raise ValueError("Target size ({}) must be the same as input size ({})"
                             .format(target.size(), input.size()))

        max_val = (-input).clamp(min=0)
        loss = input - input * target + max_val + \
               ((-max_val).exp() + (-input - max_val).exp()).log()

        invprobs = F.logsigmoid(-input * (target * 2.0 - 1.0))
        loss = (invprobs * self.gamma).exp() * loss

        return loss.sum(dim=1).mean()


def make_one_hot(labels, C=2):
    one_hot = torch.cuda.FloatTensor(labels.size(0), C, labels.size(2), labels.size(3)).zero_()
    target = one_hot.scatter_(1, labels.data, 1)

    target = Variable(target)

    return target


class FocalLossMultiLabel(torch.nn.Module):
    def __init__(self, gamma, weight):
        super().__init__()
        self.gamma = gamma
        self.nll = torch.nn.NLLLoss(weight=weight, reduce=False)

    def forward(self, input, target):
        loss = self.nll(input, target)

        one_hot = make_one_hot(target.unsqueeze(dim=1), input.size()[1])
        inv_probs = 1 - input.exp()
        focal_weights = (inv_probs * one_hot).sum(dim=1) ** self.gamma
        loss = loss * focal_weights

        return loss.mean()


class HistogramLoss(torch.nn.Module):
    """
           Evgeniya Ustinova, Victor Lempitsky
           \textit{Learning Deep Embeddings with Histogram Loss}.
           2016
    https://arxiv.org/abs/1611.00822
    """

    def __init__(self, num_steps, use_gpu=True):
        super(HistogramLoss, self).__init__()
        self.step = 2 / (num_steps - 1)  # step size
        self.use_gpu = use_gpu
        self.t = torch.range(-1, 1, self.step).view(-1, 1)
        self.t_size = self.t.size()[0]
        if self.use_gpu:
            self.t = self.t.cuda()

    def histogram(self, indices, size, s):
        s_repeat = s.repeat(self.t_size, 1)
        delta_repeat = (torch.floor((s_repeat.data + 1) / self.step) * self.step - 1).float()
        indsa = (delta_repeat == (self.t - self.step)) & indices
        indsb = (delta_repeat == self.t) & indices
        s_repeat[~(indsb | indsa)] = 0
        s_repeat[indsa] = (s_repeat - Variable(self.t) + self.step)[indsa] / self.step
        s_repeat[indsb] = (-s_repeat + Variable(self.t) + self.step)[indsb] / self.step
        return s_repeat.sum(1) / size.float()

    def forward(self, input_1, input_2, target_1, target_2):
        with torch.cuda.device(self.config['gpu']):
            target_size = target_1.size()[0]
            print('target_size ', target_size)
            print('target_1 ', target_1, target_1.shape)

            # signs_matrix = (target_1.repeat(target_size, 1) == target_2.view(-1, 1).repeat(1, target_size)).data
            signs_matrix = ((target_1.repeat(target_size, 1)* target_2.view(-1, 1).repeat(1, target_size)).sum() > 0.0).data
            print('signs_matrix ', signs_matrix, signs_matrix.shape)

            cosine_similarities = torch.mm(input_1, input_2.transpose(0, 1))

            # we want to take just upper triangle of the all pairs matrix
            # to compute the loss
            # because pairs in the lower triangle are the same

            s_indices = torch.triu(torch.ones(cosine_similarities.size()), 1).byte()

            positive_indices = signs_matrix[s_indices].repeat(self.t_size, 1)
            negative_indices = ~signs_matrix[s_indices].repeat(self.t_size, 1)

            positive_size = signs_matrix[s_indices].sum()
            negative_size = (~signs_matrix[s_indices]).sum()

            s = cosine_similarities[s_indices].view(1, -1)  # unroll cosine_similarities in a row

            histogram_positive = self.histogram(positive_indices, positive_size, s)
            histogram_negative = self.histogram(negative_indices, negative_size, s)
            histogram_positive_repeat = histogram_positive.view(-1, 1).repeat(1, histogram_positive.size()[0])
            histogram_positive_indices = torch.tril(torch.ones(histogram_positive_repeat.size()), -1).byte()

            histogram_positive_repeat[histogram_positive_indices] = 0
            histogram_positive_cdf = histogram_positive_repeat.sum(0)

            loss = torch.sum(histogram_negative * histogram_positive_cdf)

            return loss


def histogram_loss(input, target):
    inputs_number = input.shape[0]
    index_to_test = np.random.randint(low=0, high=inputs_number)
    assert torch.abs(torch.norm(input[index_to_test]) - 1.0) < 10e-5, \
        'Histogram loss should work with normalized data! ' \
        'But torch.norm(input_1[%d]) = %.10e' % (index_to_test, torch.norm(input[index_to_test]))

    loss = HistogramLoss(num_steps=150)
    return loss(input, input, target, target)
