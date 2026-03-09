"""
Sequence embedding models for protein representation.

This module provides wrappers for protein language models (ESM-2)
used to generate sequence embeddings for stoichiometry prediction.
"""

from abc import ABC, abstractmethod
from typing import List, Optional, Tuple, Union

import torch
from loguru import logger
from peft import LoraConfig, TaskType, get_peft_model, prepare_model_for_kbit_training
from torch import nn
from transformers import AutoModel, AutoTokenizer, BitsAndBytesConfig


class SeqEmbModel(ABC, nn.Module):
    """Abstract base class for sequence embedding backbones."""

    def __init__(
        self,
        model_name: str,
        max_seq_len: int = 512,
        full_length_inference: bool = False,
        max_inference_seq_len: Optional[int] = None,
        finetune: bool = False,
        load_in_4bit: bool = False,
        lora_r: int = 32,
        lora_alpha: int = 16,
        lora_dropout: float = 0.05,
    ):
        """Store common sequence-encoder configuration.

        Args:
            model_name: Hugging Face model identifier.
            max_seq_len: Fixed sequence length used during training.
            full_length_inference: Use dynamic sequence length at inference time.
            max_inference_seq_len: Optional upper bound for inference token length.
            finetune: Enable gradient updates for the language model.
            load_in_4bit: Load model weights in 4-bit quantized format.
            lora_r: LoRA rank used when fine-tuning with adapters.
            lora_alpha: LoRA scaling factor.
            lora_dropout: Dropout applied inside LoRA adapters.
        """
        super().__init__()
        self.model_name = model_name
        self.max_seq_len = max_seq_len
        self.full_length_inference = full_length_inference
        self.max_inference_seq_len = max_inference_seq_len
        self.finetune = finetune
        self.load_in_4bit = load_in_4bit
        self.lora_r = lora_r
        self.lora_alpha = lora_alpha
        self.lora_dropout = lora_dropout

    @abstractmethod
    def forward(
        self,
        sequences: List[str],
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """Encode raw amino-acid sequences into token-level representations."""
        pass

    def print_trainable_parameters(self):
        """
        Prints the number of trainable parameters.
        """
        if getattr(self, "model", None) is not None:
            trainable_params = 0
            all_param = 0
            for _, param in self.model.named_parameters():
                all_param += param.numel()
                if param.requires_grad:
                    trainable_params += param.numel()
            logger.info(
                f"Model name: {self.model_name}\n"
                f"trainable params: {trainable_params} || "
                f"all params: {all_param} || "
                f"trainable (%): {100 * trainable_params / all_param:.2f}"
            )
        else:
            logger.info(f"Model name: {self.model_name} has not been initialized")


class Esm2(SeqEmbModel):
    """ESM-2 based sequence embedding model with optional LoRA fine-tuning."""

    def __init__(
        self,
        model_name: str,
        max_seq_len: int = 512,
        full_length_inference: bool = False,
        max_inference_seq_len: Optional[int] = None,
        finetune: bool = False,
        load_in_4bit: bool = False,
        lora_r: int = 32,
        lora_alpha: int = 16,
        lora_dropout: float = 0.05,
    ):
        """Initialize ESM-2 backbone, tokenizer, and training mode settings."""
        super().__init__(
            model_name,
            max_seq_len,
            full_length_inference,
            max_inference_seq_len,
            finetune,
            load_in_4bit,
            lora_r,
            lora_alpha,
            lora_dropout,
        )
        logger.info(f"Loading in 4bit: {self.load_in_4bit}")
        if self.load_in_4bit:
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=self.load_in_4bit,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.bfloat16,
            )
            self.model = AutoModel.from_pretrained(
                self.model_name,
                quantization_config=bnb_config,
                device_map="auto",
                _fast_init=False,
                add_pooling_layer=False,
            )
            self.model = prepare_model_for_kbit_training(self.model)
        else:
            self.model = AutoModel.from_pretrained(
                self.model_name,
                device_map="auto",
                add_pooling_layer=False,
            )
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)

        if not self.finetune:
            for param in self.model.parameters():
                param.requires_grad = False
            self.model.eval()
        else:
            lora_config = LoraConfig(
                r=lora_r,
                lora_alpha=lora_alpha,
                lora_dropout=lora_dropout,
                bias="none",
                target_modules=["query", "key", "value"],
                task_type=TaskType.FEATURE_EXTRACTION,
            )
            self.model = get_peft_model(self.model, lora_config)
        self.print_trainable_parameters()
        self.seq_embed_size = self.model.config.hidden_size

    def _get_inference_max_length(self, sequences: List[str]) -> int:
        """Compute the tokenizer max_length for full-length inference."""
        longest = max(len(seq) for seq in sequences) + 2  # +2 for CLS/EOS
        if self.max_inference_seq_len is not None:
            longest = min(longest, self.max_inference_seq_len)
        return longest

    def forward(
        self, sequences: List[str]
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """Generate per-token embeddings and corresponding padding mask.

        Args:
            sequences: Protein sequences as strings.

        Returns:
            A tuple ``(embeddings, attention_mask)`` where:
            - embeddings: Tensor of shape ``[B, L, D]`` without CLS/EOS tokens.
            - attention_mask: Boolean tensor ``[B, L]`` with ``True`` for padding.
        """
        if self.full_length_inference and not self.training:
            max_len = self._get_inference_max_length(sequences)
            inputs = self.tokenizer(
                sequences,
                return_tensors="pt",
                max_length=max_len,
                truncation=True,
                padding="longest",
            )
        else:
            inputs = self.tokenizer(
                sequences,
                return_tensors="pt",
                max_length=self.max_seq_len,
                truncation=True,
                padding="max_length",
            )

        inputs = {k: v.to(self.model.device) for k, v in inputs.items()}

        with torch.set_grad_enabled(self.finetune):
            outputs = self.model(**inputs)

        # out EOS/padding so only residue tokens are treated as valid.
        per_residue_embeddings = outputs.last_hidden_state[:, 1:-1]
        max_residue_tokens = per_residue_embeddings.size(1)
        seq_lens = torch.tensor(
            [len(seq) for seq in sequences],
            device=per_residue_embeddings.device,
        ).clamp(max=max_residue_tokens)
        attention_mask = (
            torch.arange(max_residue_tokens, device=per_residue_embeddings.device)
            .unsqueeze(0)
            .ge(seq_lens.unsqueeze(1))
        )

        return per_residue_embeddings, attention_mask


class NoEmb(SeqEmbModel):
    """No-embeddings egeneration case model"""

    def __init__(
        self,
        model_name: str,
        max_seq_len: int = 512,
        full_length_inference: bool = False,
        max_inference_seq_len: Optional[int] = None,
        finetune: bool = False,
        load_in_4bit: bool = False,
        lora_r: int = 32,
        lora_alpha: int = 16,
        lora_dropout: float = 0.05,
    ):
        """Initialize ESM-2 backbone, tokenizer, and training mode settings."""
        super().__init__(
            model_name,
            max_seq_len,
            full_length_inference,
            max_inference_seq_len,
            finetune,
            load_in_4bit,
            lora_r,
            lora_alpha,
            lora_dropout,
        )

        self.seq_embed_size = 1280  # Default to ESM-2 embedding size for compatibility

    def forward(
        self, sequences: List[str]
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """Generate per-token embeddings and corresponding padding mask.

        Args:
            sequences: Protein sequences as strings.

        Returns:
            A tuple ``(embeddings, attention_mask)`` where:
            - embeddings: Tensor of shape ``[B, L, D]`` without CLS/EOS tokens.
            - attention_mask: Boolean tensor ``[B, L]`` with ``True`` for padding.
        """
        return None
