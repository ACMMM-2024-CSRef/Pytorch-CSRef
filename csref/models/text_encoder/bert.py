import torch.nn as nn
from transformers import BertModel


class Bert(nn.Module):
    def __init__(
            self,
            pretrained_path="data/weights/bert-base-uncased",
            freeze_model=True,
            hidden_state_index=-1,
    ):
        super(Bert, self).__init__()

        self.hidden_state_index = hidden_state_index

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

        hidden_state = output.hidden_states[self.hidden_state_index]
        feat = hidden_state[:, 0, :]  # corresponding to [CLS] token

        return feat
