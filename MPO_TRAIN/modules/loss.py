import torch
import torch.nn as nn


class LanguageModelCriterion(nn.Module):
    def __init__(self):
        super(LanguageModelCriterion, self).__init__()

    def forward(self, input, target, mask):
        # truncate to the same size
        target = target[:, :input.size(1)]
        mask = mask[:, :input.size(1)]
        output = -input.gather(2, target.long().unsqueeze(2)).squeeze(2) * mask
        output = torch.sum(output) / torch.sum(mask)
        return output


class LossWrapper(nn.Module):
    def __init__(self):
        super(LossWrapper, self).__init__()
        self.criterion = LanguageModelCriterion()
        self.criterion_mlc = nn.BCELoss()

    def forward(self, output, output_mlc, reports_ids, reports_masks, label):
        loss = self.criterion(output, reports_ids[:, 1:], reports_masks[:, 1:]).mean()
        loss_mlc = self.criterion_mlc(output_mlc, label)
        return loss + loss_mlc


def compute_loss(output, reports_ids, reports_masks):
    criterion = LanguageModelCriterion()
    loss = criterion(output, reports_ids[:, 1:], reports_masks[:, 1:]).mean()
    return loss


class RewardCriterion(nn.Module):
    def __init__(self):
        super(RewardCriterion, self).__init__()

    def forward(self, input, seq, reward):
        # print("sample_log:" , input.shape)
        # print('seq_shape:' , seq.shape)
        # print('reward_shape:', reward.shape)
        # seq_length = input.shape[1]
        # embedding_layer = nn.Linear(2, 480).to(input.device)
        # preference = torch.tensor([hyp], dtype=torch.float32, device=input.device)  # 加上一个维度
        #
        # # 通过嵌入层进行嵌入
        # embedded_preference = embedding_layer(preference).to(input.device)
        # embedded_preference = embedded_preference.view(8, 60).to(input.device)
        input = input.gather(2, seq.unsqueeze(2)).squeeze(2)

        input = input.reshape(-1)
        reward = reward.reshape(-1)
        mask = (seq > 0).to(input)
        mask = torch.cat([mask.new(mask.size(0), 1).fill_(1), mask[:, :-1]], 1).reshape(-1)
        output = - input * reward  * mask
        output = torch.sum(output) / torch.sum(mask)

        return output
        # input_gathered = input.gather(2, seq.unsqueeze(2)).squeeze(2)

        # 考虑到序列中每个位置的重要性
        # mask = (seq > 0).float()  # 确保mask是float类型
        # # 创建一个新的mask，第一个位置为1，其余位置保持不变
        # mask_shifted = torch.cat([torch.ones(mask.size(0), 1, device=mask.device), mask[:, :-1]], 1)
        # # 假设 reward 的原始形状为 [batch_size]
        # reward_expanded = reward.unsqueeze(1) # 添加两个维度，形状变为 [batch_size, 1, 1]
        # reward_expanded = reward_expanded.expand(-1, -1, seq_length)  # 扩展到正确的形状 [batch_size, 1, seq_length]
        #
        # # 计算所有输入对数概率的负和
        # all_logprobs_neg_sum = torch.sum(-input_gathered * mask_shifted, dim=1)
        # # reward_expanded = reward.unsqueeze(1).expand(-1, seq_length)  # 现在 reward 的形状是 [8, 60]
        #
        # # 现在可以安全地进行元素级乘法操作
        # reward_loss = reward_expanded * all_logprobs_neg_sum
        # # 计算基于奖励的损失
        # # reward_loss = (reward) * all_logprobs_neg_sum
        # reward_loss = reward_loss.mean()
        # 假设 reward 的形状是 [batch_size]，例如 [8]
        # all_logprobs_neg_sum 的形状是 [batch_size, seq_length]，例如 [8, 60]

        # # 扩展 reward 以匹配 all_logprobs_neg_sum 的形状


        # 根据需要进一步处理 reward_loss
        # input_neg_log_prob = -torch.gather(input, 2, seq.unsqueeze(2)).squeeze(2)
        #
        # # 计算奖励差异
        # reward_diff = reward
        #
        # # 应用掩码，忽略seq中的0值（假设序列用0填充）
        # mask = (seq > 0).float()  # 确保使用float型掩码以匹配input的数据类型
        #
        # # 计算加权的负对数概率和
        # weighted_neg_log_prob_sum = torch.sum(input_neg_log_prob * mask, dim=1)
        # weighted_neg_log_prob_sum = weighted_neg_log_prob_sum.unsqueeze(1)
        #
        # # 计算最终的损失
        # reward_loss = reward_diff * weighted_neg_log_prob_sum
        # reward_loss = reward_loss.mean()
        # return reward_loss

