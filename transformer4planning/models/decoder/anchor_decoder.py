import torch
from torch import nn
from transformer4planning.libs.mlp import DecoderResCat
from torch.nn import CrossEntropyLoss, MSELoss
    
class AnchorDecoder(nn.Module):
    def __init__(self, model_args, config, anchor_num=64, anchor_len=1):
        super().__init__()
        self.model_args = model_args
        self.anchor_num = anchor_num
        self.anchor_len = anchor_len
        self.anchor_cls_decoder = DecoderResCat(config.n_inner, config.n_embd, out_features=self.anchor_num)
        self.anchor_logits_decoder = DecoderResCat(config.n_inner, config.n_embd, out_features=2 * self.anchor_num)

        self.cls_anchor_loss = CrossEntropyLoss(reduction="none")
        self.logits_anchor_loss = MSELoss(reduction="none")

    def compute_anchor_loss(self, 
                          hidden_output,
                          info_dict):
        """
        pred future 8-s trajectory and compute loss(l2 loss or smooth l1)
        params:
            hidden_output: whole hidden state output from transformer backbone
            label: ground truth trajectory in future 8-s
            info_dict: dict contains additional infomation, such as context length/input length, pred length, etc. 
        """
        gt_anchor_cls= info_dict["gt_anchor_cls"]
        gt_anchor_mask = info_dict["gt_anchor_mask"]
        gt_anchor_logits= info_dict["gt_anchor_logits"]

        context_length = info_dict["context_length"] * 2
        pred_anchor_embed = hidden_output[:, context_length-1:context_length-1+self.anchor_len, :] # (bs, anchor_len, n_embed)
        pred_anchor_cls = self.anchor_cls_decoder(pred_anchor_embed) # (bs, anchor_len, 64)
        pred_anchor_logits = self.anchor_logits_decoder(pred_anchor_embed)
        loss_anchor = self.cls_anchor_loss(pred_anchor_cls.reshape(-1, self.anchor_num).to(torch.float64), gt_anchor_cls.reshape(-1).long())
        loss_anchor = (loss_anchor * gt_anchor_mask.view(-1)).sum()/ (gt_anchor_mask.sum()+1e-7)
        
        bs = gt_anchor_cls.shape[0]
        pred_anchor_logits = pred_anchor_logits.view(bs, self.anchor_num, 2)
        
        pred_pos_anchor_logits = pred_anchor_logits[torch.arange(bs), gt_anchor_cls, :] # (bs, 2)            
        loss_anchor_logits = self.logits_anchor_loss(pred_pos_anchor_logits, gt_anchor_logits)
        loss_anchor_logits = (loss_anchor_logits * gt_anchor_mask).sum() / (gt_anchor_mask.sum() + 1e-7)

        info_dict["input_length"] = context_length + self.anchor_len
        del info_dict["context_length"]
        
        return loss_anchor, loss_anchor_logits