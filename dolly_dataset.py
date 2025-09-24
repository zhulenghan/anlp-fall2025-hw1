import torch
from torch.utils.data import Dataset, DataLoader
from typing import Optional, List, Dict
import json
import random
from huggingface_hub import hf_hub_download
from tokenizer import Tokenizer

# Instruction prompt template (Alpaca-style)
PROMPT_TEMPLATE = """Below is an instruction that describes a task. Write a response that appropriately completes the request.

### Instruction:
{instruction}

### Response:
{response}"""

PROMPT_TEMPLATE_NO_INPUT = """Below is an instruction that describes a task. Write a response that appropriately completes the request.

### Instruction:
{instruction}

### Response:
"""

class DollyDataset(Dataset):
    def __init__(
        self,
        tokenizer: Tokenizer,
        max_length: Optional[int] = None,  # Optional max length for filtering
        split: str = "train",
    ):
        """
        Dataset for Databricks Dolly-15k instruction tuning.

        Args:
            tokenizer: Tokenizer instance
            max_length: Optional maximum sequence length for filtering out too-long samples
            split: train/validation split
        """
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.split = split

        # Load data (always from Hugging Face)
        self.data = self.load_dolly_data()

        # Split data (90% train, 10% validation)
        random.seed(42)
        indices = list(range(len(self.data)))
        random.shuffle(indices)

        split_idx = int(len(indices) * 0.9)
        if split == "train":
            self.indices = indices[:split_idx]
        else:
            self.indices = indices[split_idx:]

        print(f"Loaded {len(self.indices)} {split} samples from Dolly-15k dataset")

    def load_dolly_data(self) -> List[Dict]:
        """Load Dolly-15k dataset from Hugging Face hub."""
        print("Downloading Dolly-15k dataset via huggingface_hub...")
        file_path = hf_hub_download(
            repo_id="databricks/databricks-dolly-15k",
            filename="databricks-dolly-15k.jsonl",
            repo_type="dataset"
        )
        print(f"Dataset downloaded to: {file_path}")

        # Load JSONL file
        data = []
        with open(file_path, 'r') as f:
            for line in f:
                data.append(json.loads(line))

        # Process data into instruction-response pairs
        processed_data = []
        for item in data:
            instruction = item.get("instruction", "")
            context = item.get("context", "")
            response = item.get("response", "")

            # Combine instruction and context if context exists
            if context and context.strip():
                full_instruction = f"{instruction}\n\nContext: {context}"
            else:
                full_instruction = instruction

            if full_instruction.strip() and response.strip():
                processed_data.append({
                    "instruction": full_instruction.strip(),
                    "response": response.strip()
                })

        return processed_data

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        data_idx = self.indices[idx]
        item = self.data[data_idx]

        # Format prompt only
        prompt = PROMPT_TEMPLATE_NO_INPUT.format(
            instruction=item["instruction"]
        )

        # Tokenize prompt and response separately to get correct boundaries
        prompt_ids = self.tokenizer.encode(prompt, bos=True, eos=False)
        response_ids = self.tokenizer.encode(item["response"], bos=False, eos=True)

        # Combine
        input_ids = prompt_ids + response_ids

        # Create labels: -100 for prompt tokens (no loss), actual ids for response
        labels = [-100] * len(prompt_ids) + response_ids

        # Deterministic truncation if too long
        if self.max_length and len(input_ids) > self.max_length:
            input_ids = input_ids[:self.max_length]
            labels = labels[:self.max_length]
            # Ensure EOS token at end
            input_ids[-1] = self.tokenizer.eos_id
            labels[-1] = self.tokenizer.eos_id

        return {
            "input_ids": input_ids,
            "labels": labels,
            "instruction": item["instruction"],
            "response": item["response"]
        }

    def collate_fn(self, batch):
        """Dynamic padding collate function following ChatGPT's approach."""
        # Find max length in this batch
        max_len = max(len(item["input_ids"]) for item in batch)

        input_ids = []
        attention_masks = []
        labels = []

        for item in batch:
            ids = item["input_ids"]
            item_labels = item["labels"]

            # Padding
            padding_len = max_len - len(ids)

            # Pad input_ids with pad_token_id
            padded_ids = ids + [self.tokenizer.pad_id] * padding_len

            # Pad labels with -100 (ignored by loss)
            padded_labels = item_labels + [-100] * padding_len

            # Attention mask (1 for real tokens, 0 for padding)
            attention_mask = [1] * len(ids) + [0] * padding_len

            input_ids.append(padded_ids)
            attention_masks.append(attention_mask)
            labels.append(padded_labels)

        return {
            "input_ids": torch.LongTensor(input_ids),
            "attention_mask": torch.LongTensor(attention_masks),
            "labels": torch.LongTensor(labels)
        }


def create_dolly_dataloaders(
    tokenizer: Tokenizer,
    batch_size: int = 4,
    max_length: Optional[int] = None,
):
    """Create train and validation dataloaders for Dolly dataset (Hugging Face only)."""

    train_dataset = DollyDataset(
        tokenizer=tokenizer,
        max_length=max_length,
        split="train",
    )

    val_dataset = DollyDataset(
        tokenizer=tokenizer,
        max_length=max_length,
        split="validation",
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=train_dataset.collate_fn
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=val_dataset.collate_fn
    )

    return train_loader, val_loader
