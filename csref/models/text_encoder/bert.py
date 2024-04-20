import torch.nn as nn
from transformers import BertModel


class Bert(nn.Module):
    def __init__(
            self,
            pretrained_path="data/weights/bert-large-uncased",
            freeze_model=True,
            use_one_hidden_state_as_feat=True,
            hidden_state_index=-1,
            use_att_flat_mask=True,
    ):
        super(Bert, self).__init__()

        self.use_one_hidden_state_as_feat = use_one_hidden_state_as_feat
        self.hidden_state_index = hidden_state_index
        self.use_att_flat_mask = use_att_flat_mask

        self.model = BertModel.from_pretrained(pretrained_path)

        if freeze_model:
            self.frozen(self.model)

    def frozen(self, module):
        if getattr(module, 'module', False):
            for child in module.module():
                for param in child.parameters():
                    param.requires_grad = False
        else:
            for param in module.parameters():
                param.requires_grad = False

    def forward(self, text_ids, mask):

        output = self.model(
            # input_values=text,
            input_ids=text_ids,
            attention_mask=mask,
            output_attentions=True,
            output_hidden_states=True,
            return_dict=True
        )

        feat = None
        if self.use_one_hidden_state_as_feat:
            hidden_state = output.hidden_states[self.hidden_state_index]  # large[b, len ,c(1024)]
            feat = hidden_state[:, 0, :]  # (batch, len, channel)
        return feat
