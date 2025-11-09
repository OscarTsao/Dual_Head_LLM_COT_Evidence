"""
Dual-Head Reasoning Distillation (DHRD) Model

Implementation of the paper:
"Dual-Head Reasoning Distillation: Improving Classifier Accuracy with Train-Time-Only Reasoning"
arXiv: 2509.21487

This module implements a dual-head architecture for decoder-only language models that:
1. Uses a classification head (pooled) for both training and inference
2. Uses a reasoning head (LM head) only during training for distillation
3. Achieves improved classification accuracy with fast inference
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
from typing import Optional, Tuple, Dict, Any
from dataclasses import dataclass


@dataclass
class DHRDOutput:
    """Output class for DHRD model"""
    loss: Optional[torch.FloatTensor] = None
    classification_loss: Optional[torch.FloatTensor] = None
    reasoning_loss: Optional[torch.FloatTensor] = None
    classification_logits: Optional[torch.FloatTensor] = None
    reasoning_logits: Optional[torch.FloatTensor] = None
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    attentions: Optional[Tuple[torch.FloatTensor]] = None


class ClassificationHead(nn.Module):
    """
    Classification head with last-token pooling.
    Used during both training and inference.
    """

    def __init__(
        self,
        hidden_size: int,
        num_labels: int,
        dropout_prob: float = 0.1,
        pooling_method: str = "last"
    ):
        """
        Args:
            hidden_size: Dimension of hidden states from transformer
            num_labels: Number of classification labels
            dropout_prob: Dropout probability
            pooling_method: Method for pooling ('last' for last-token pooling)
        """
        super().__init__()
        self.pooling_method = pooling_method
        self.dropout = nn.Dropout(dropout_prob)
        self.classifier = nn.Linear(hidden_size, num_labels)

    def pool_hidden_states(
        self,
        hidden_states: torch.FloatTensor,
        attention_mask: Optional[torch.LongTensor] = None
    ) -> torch.FloatTensor:
        """
        Pool hidden states using specified method.

        Args:
            hidden_states: [batch_size, seq_len, hidden_size]
            attention_mask: [batch_size, seq_len]

        Returns:
            pooled_states: [batch_size, hidden_size]
        """
        if self.pooling_method == "last":
            # Last-token pooling: get the last non-padding token
            if attention_mask is not None:
                # Find the last non-padding token for each sequence
                sequence_lengths = attention_mask.sum(dim=1) - 1
                batch_size = hidden_states.shape[0]
                pooled_states = hidden_states[
                    torch.arange(batch_size, device=hidden_states.device),
                    sequence_lengths
                ]
            else:
                # If no attention mask, use the last token
                pooled_states = hidden_states[:, -1, :]
        elif self.pooling_method == "mean":
            # Mean pooling over sequence length
            if attention_mask is not None:
                mask_expanded = attention_mask.unsqueeze(-1).expand(hidden_states.size()).float()
                sum_hidden = torch.sum(hidden_states * mask_expanded, dim=1)
                sum_mask = torch.clamp(mask_expanded.sum(dim=1), min=1e-9)
                pooled_states = sum_hidden / sum_mask
            else:
                pooled_states = hidden_states.mean(dim=1)
        else:
            raise ValueError(f"Unknown pooling method: {self.pooling_method}")

        return pooled_states

    def forward(
        self,
        hidden_states: torch.FloatTensor,
        attention_mask: Optional[torch.LongTensor] = None
    ) -> torch.FloatTensor:
        """
        Forward pass of classification head.

        Args:
            hidden_states: [batch_size, seq_len, hidden_size]
            attention_mask: [batch_size, seq_len]

        Returns:
            logits: [batch_size, num_labels]
        """
        pooled_states = self.pool_hidden_states(hidden_states, attention_mask)
        pooled_states = self.dropout(pooled_states)
        logits = self.classifier(pooled_states)
        return logits


class ReasoningHead(nn.Module):
    """
    Reasoning head (LM head) for generating rationales.
    Used only during training for distillation.
    """

    def __init__(self, hidden_size: int, vocab_size: int):
        """
        Args:
            hidden_size: Dimension of hidden states from transformer
            vocab_size: Size of vocabulary
        """
        super().__init__()
        self.lm_head = nn.Linear(hidden_size, vocab_size, bias=False)

    def forward(self, hidden_states: torch.FloatTensor) -> torch.FloatTensor:
        """
        Forward pass of reasoning head.

        Args:
            hidden_states: [batch_size, seq_len, hidden_size]

        Returns:
            logits: [batch_size, seq_len, vocab_size]
        """
        return self.lm_head(hidden_states)


class DHRDModel(nn.Module):
    """
    Dual-Head Reasoning Distillation Model

    A decoder-only transformer with two heads:
    1. Classification head (used in training and inference)
    2. Reasoning head (used only in training)
    """

    def __init__(
        self,
        backbone_model_name: str,
        num_labels: int,
        classification_dropout: float = 0.1,
        pooling_method: str = "last",
        reasoning_weight: float = 0.5,
        use_pretrained: bool = True,
        **kwargs
    ):
        """
        Args:
            backbone_model_name: Name of pretrained decoder-only model (e.g., 'gpt2', 'facebook/opt-125m')
            num_labels: Number of classification labels
            classification_dropout: Dropout for classification head
            pooling_method: Pooling method for classification ('last' or 'mean')
            reasoning_weight: Weight for reasoning loss in combined loss (lambda)
            use_pretrained: Whether to use pretrained weights
        """
        super().__init__()

        self.num_labels = num_labels
        self.reasoning_weight = reasoning_weight
        self.training_mode = True  # Flag to enable/disable reasoning head

        # Load backbone decoder-only transformer
        if use_pretrained:
            self.backbone = AutoModelForCausalLM.from_pretrained(
                backbone_model_name,
                **kwargs
            )
        else:
            config = AutoConfig.from_pretrained(backbone_model_name)
            self.backbone = AutoModelForCausalLM.from_config(config)

        # Get the base model (without LM head)
        if hasattr(self.backbone, 'model'):
            self.transformer = self.backbone.model
        elif hasattr(self.backbone, 'transformer'):
            self.transformer = self.backbone.transformer
        elif hasattr(self.backbone, 'gpt_neox'):
            self.transformer = self.backbone.gpt_neox
        else:
            # Fallback: use the backbone itself
            self.transformer = self.backbone

        hidden_size = self.backbone.config.hidden_size
        vocab_size = self.backbone.config.vocab_size

        # Classification head (used in training and inference)
        self.classification_head = ClassificationHead(
            hidden_size=hidden_size,
            num_labels=num_labels,
            dropout_prob=classification_dropout,
            pooling_method=pooling_method
        )

        # Reasoning head (used only in training)
        # Reuse the LM head from backbone if available
        if hasattr(self.backbone, 'lm_head'):
            self.reasoning_head = self.backbone.lm_head
        else:
            self.reasoning_head = ReasoningHead(hidden_size, vocab_size)

    def enable_reasoning_head(self):
        """Enable reasoning head for training"""
        self.training_mode = True

    def disable_reasoning_head(self):
        """Disable reasoning head for inference"""
        self.training_mode = False

    def forward(
        self,
        input_ids: torch.LongTensor,
        attention_mask: Optional[torch.LongTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        rationale_input_ids: Optional[torch.LongTensor] = None,
        rationale_attention_mask: Optional[torch.LongTensor] = None,
        rationale_labels: Optional[torch.LongTensor] = None,
        return_dict: bool = True,
        output_hidden_states: bool = False,
        output_attentions: bool = False,
    ) -> DHRDOutput:
        """
        Forward pass of DHRD model.

        Args:
            input_ids: Input token IDs [batch_size, seq_len]
            attention_mask: Attention mask [batch_size, seq_len]
            labels: Classification labels [batch_size]
            rationale_input_ids: Input IDs with rationales [batch_size, seq_len_with_rationale]
            rationale_attention_mask: Attention mask for rationales
            rationale_labels: Token labels for LM loss [batch_size, seq_len_with_rationale]
            return_dict: Whether to return DHRDOutput
            output_hidden_states: Whether to output hidden states
            output_attentions: Whether to output attentions

        Returns:
            DHRDOutput containing losses and logits
        """
        # Forward pass through transformer for classification
        outputs = self.transformer(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=output_hidden_states,
            output_attentions=output_attentions,
            return_dict=True
        )

        hidden_states = outputs.last_hidden_state

        # Classification head forward pass
        classification_logits = self.classification_head(
            hidden_states=hidden_states,
            attention_mask=attention_mask
        )

        # Initialize outputs
        total_loss = None
        classification_loss = None
        reasoning_loss = None
        reasoning_logits = None

        # Compute classification loss
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            classification_loss = loss_fct(
                classification_logits.view(-1, self.num_labels),
                labels.view(-1)
            )
            total_loss = classification_loss

        # Reasoning head forward pass (only during training)
        if self.training_mode and rationale_input_ids is not None:
            # Forward pass for rationale sequence
            rationale_outputs = self.transformer(
                input_ids=rationale_input_ids,
                attention_mask=rationale_attention_mask,
                output_hidden_states=output_hidden_states,
                return_dict=True
            )

            rationale_hidden_states = rationale_outputs.last_hidden_state
            reasoning_logits = self.reasoning_head(rationale_hidden_states)

            # Compute reasoning loss (token-level LM loss)
            if rationale_labels is not None:
                # Shift logits and labels for causal LM
                shift_logits = reasoning_logits[..., :-1, :].contiguous()
                shift_labels = rationale_labels[..., 1:].contiguous()

                loss_fct = nn.CrossEntropyLoss(ignore_index=-100)
                reasoning_loss = loss_fct(
                    shift_logits.view(-1, shift_logits.size(-1)),
                    shift_labels.view(-1)
                )

                # Combine losses: total_loss = classification_loss + lambda * reasoning_loss
                if total_loss is not None:
                    total_loss = (1 - self.reasoning_weight) * classification_loss + \
                                 self.reasoning_weight * reasoning_loss
                else:
                    total_loss = reasoning_loss

        if not return_dict:
            output = (classification_logits, reasoning_logits)
            if total_loss is not None:
                output = (total_loss,) + output
            return output

        return DHRDOutput(
            loss=total_loss,
            classification_loss=classification_loss,
            reasoning_loss=reasoning_loss,
            classification_logits=classification_logits,
            reasoning_logits=reasoning_logits,
            hidden_states=outputs.hidden_states if output_hidden_states else None,
            attentions=outputs.attentions if output_attentions else None,
        )

    def predict(
        self,
        input_ids: torch.LongTensor,
        attention_mask: Optional[torch.LongTensor] = None,
    ) -> torch.FloatTensor:
        """
        Fast inference mode (reasoning head disabled).

        Args:
            input_ids: Input token IDs [batch_size, seq_len]
            attention_mask: Attention mask [batch_size, seq_len]

        Returns:
            logits: Classification logits [batch_size, num_labels]
        """
        # Ensure reasoning head is disabled
        was_training = self.training_mode
        self.disable_reasoning_head()

        with torch.no_grad():
            outputs = self.transformer(
                input_ids=input_ids,
                attention_mask=attention_mask,
                return_dict=True
            )

            hidden_states = outputs.last_hidden_state
            logits = self.classification_head(
                hidden_states=hidden_states,
                attention_mask=attention_mask
            )

        # Restore training mode
        if was_training:
            self.enable_reasoning_head()

        return logits

    def get_num_parameters(self, only_trainable: bool = False) -> int:
        """
        Get the number of parameters in the model.

        Args:
            only_trainable: If True, count only trainable parameters

        Returns:
            Number of parameters
        """
        if only_trainable:
            return sum(p.numel() for p in self.parameters() if p.requires_grad)
        else:
            return sum(p.numel() for p in self.parameters())
