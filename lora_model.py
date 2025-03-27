import transformers
import torch.nn as nn
import torch
import torch.nn.functional as F
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from peft import get_peft_model, LoraConfig, TaskType


class LoRA(nn.Module):
    def __init__(self, linear, input_dim, output_dim, ranks = [8,16,32,64], topk = 1, alpha = 4):
        super().__init__()
        self.ranks = sorted(ranks)  
        max_rank = self.ranks[-1]

        self.B = nn.Parameter(torch.zeros(output_dim, max_rank))  
        self.A = nn.Parameter(torch.randn(max_rank, input_dim) * 0.02)  

        self.routing_gate = nn.Linear(input_dim, len(ranks))
        self.linear = linear
        self.topk = topk
        self.scaling = 1/alpha

    def forward(self, hidden_states):
        batch_size, seq_length, hidden_dim = hidden_states.shape
        out = self.linear(hidden_states)

        # Reshape for per-token processing
        hidden_states_flat = hidden_states.view(-1, hidden_dim)

        # Compute routing weights
        routing_scores = self.routing_gate(hidden_states_flat)
        routing_weights = F.softmax(routing_scores, dim=-1)
        routing_weights, selected_experts = torch.topk(routing_weights, self.topk, dim=-1)
        routing_weights = routing_weights / routing_weights.sum(dim=-1, keepdim=True)

        # Precompute all possible LoRA outputs
        lora_outputs = []
        for rank in self.ranks:
            A_rank = self.A[:rank, :] 
            B_rank = self.B[:, :rank]  
            lora_out = F.linear(F.linear(hidden_states_flat, A_rank), B_rank) # (batch*seq, output_dim)
            lora_outputs.append(lora_out)
        
        # Stack and gather
        lora_outputs = torch.stack(lora_outputs, dim=1)  # (batch*seq, num_ranks, output_dim)
        selected_outputs = torch.gather(
            lora_outputs, 
            dim=1, 
            index=selected_experts.unsqueeze(-1).expand(-1, -1, lora_outputs.shape[-1])
        )  # (batch*seq, topk, output_dim)
        
        # Weighted sum
        weighted_outputs = selected_outputs * routing_weights.unsqueeze(-1)
        final_hidden_states = weighted_outputs.sum(dim=1)  # (batch*seq, output_dim)
        
        # Reshape and add to original output
        final_hidden_states = final_hidden_states.view(batch_size, seq_length, -1)
        out += final_hidden_states
        return out

def replace_linear_layers(module, custom_linear_cls, ranks):
    for name, child in module.named_children():
        
        if isinstance(child, nn.Linear):
            # print(name,child)
            if name in ["value", "query"]:
                new_layer = LoRA(child, child.in_features, child.out_features, ranks)
                setattr(module, name, new_layer)
        else:
        
            replace_linear_layers(child, custom_linear_cls, ranks)

def get_matlora(model, ranks):
    before_lora = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print("Before LoRA:", before_lora)
    
    for p in model.parameters():
        p.requires_grad = False
    replace_linear_layers(model, LoRA, ranks)
    
    after_lora = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print("After LoRA:", after_lora)

    print("Percentage of Trainable Parameters: ", (after_lora / before_lora * 100).__round__(3))
    return model
    

def get_lora(model):

    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM ,  # Sequence Classification
        inference_mode=False,  # Enable training
        r = 64,  # Rank (Lower rank = more efficiency)
        lora_alpha = 64,  # Scaling factor
        lora_dropout=0.0,  # Dropout probability
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "down_proj", "up_proj"],
        # exclude_modules=["lm_head"]
    )
    before_lora = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print("Before LoRA:", before_lora)
    model = get_peft_model(model, lora_config)

    after_lora = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print("After LoRA:", after_lora)


    return model



