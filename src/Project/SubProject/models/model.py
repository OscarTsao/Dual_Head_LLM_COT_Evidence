import torch
import transformers

class classification_head():
    def __init__(self, input_dim: int, num_labels: int, dropout_prob: float = 0.1, layer_num: int = 1):
        self.linear = torch.nn.Linear(input_dim, num_labels)

    def forward(self, x):
        return self.linear(x)

class Model(torch.nn.Module):
    def __init__(self, model_name: str, num_labels: int):
        super(Model, self).__init__()
        self.transformer = transformers.AutoModel.from_pretrained(model_name)
        self.classifier = torch.nn.Linear(self.transformer.config.hidden_size, num_labels)

    def forward(self, input_ids, attention_mask):
        outputs = self.transformer(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs[1]  # Assuming the second output is the pooled output
        logits = self.classifier(pooled_output)
        return logits

