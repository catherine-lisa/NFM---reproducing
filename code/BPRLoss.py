import torch
from torch import nn
import numpy as np

class BPRLoss:
    def __init__(self, model, decay, lr, opt):
        self.model = model
        self.weight_decay = decay
        self.lr = lr
        self.opt = opt

    def stageOne(self, users, pos, neg):
        # 计算出bpr_loss的值
        loss, reg_loss = self.bpr_loss(users, pos, neg)
        reg_loss = reg_loss*self.weight_decay
        loss = loss + reg_loss
        # 清空上一次的梯度记录
        self.opt.zero_grad()
        #  PyTorch的反向传播(即tensor.backward())是通过autograd包来实现的，autograd包会根据tensor进行过的数学运算来自动计算其对应的梯度。
        loss.backward()
        # step()函数的作用是执行一次优化步骤，通过梯度下降法来更新参数的值。
        self.opt.step()
        return loss.cpu().item()

    def bpr_loss(self, users, pos, neg):
        # 值得注意的是，原BPR就是让正样本和负样本的得分之差尽可能达到最大
        # 这里没有负号，是尽可能的小
        (users_emb, pos_emb, neg_emb, 
        userEmb0,  posEmb0, negEmb0) = self.getEmbedding(users.long(), pos.long(), neg.long())
        reg_loss = (1/2)*(userEmb0.norm(2).pow(2) + 
                         posEmb0.norm(2).pow(2)  +
                         negEmb0.norm(2).pow(2))/float(len(users))
        pos_scores = torch.mul(users_emb, pos_emb)
        pos_scores = torch.sum(pos_scores, dim=1)
        neg_scores = torch.mul(users_emb, neg_emb)
        neg_scores = torch.sum(neg_scores, dim=1)
        loss = torch.mean(torch.nn.functional.softplus(neg_scores - pos_scores))
        
        return loss, reg_loss