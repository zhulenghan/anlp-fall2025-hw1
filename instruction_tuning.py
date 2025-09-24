import argparse
import math
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import os
import time
from tqdm import tqdm
import json
from typing import Optional

from llama import Llama, load_pretrained
from tokenizer import Tokenizer
from optimizer import AdamW
from dolly_dataset import create_dolly_dataloaders, PROMPT_TEMPLATE_NO_INPUT


def compute_loss(logits, labels):
    """
    Compute cross-entropy loss using -100 masking (standard PyTorch approach).

    Args:
        logits: (batch_size, seq_len, vocab_size)
        labels: (batch_size, seq_len) with -100 for ignored positions
    """
    batch_size, seq_len, vocab_size = logits.shape

    # Shift for next-token prediction
    shift_logits = logits[:, :-1, :].contiguous()
    shift_labels = labels[:, 1:].contiguous()

    # Flatten for loss computation
    shift_logits = shift_logits.view(-1, vocab_size)
    shift_labels = shift_labels.view(-1)

    # CrossEntropyLoss automatically ignores -100 labels
    loss = F.cross_entropy(shift_logits, shift_labels, ignore_index=-100)

    return loss


def train_epoch(model, dataloader, optimizer, device, accumulation_steps=1, scheduler=None):
    """Train for one epoch."""
    model.train()
    total_loss = 0
    num_batches = 0

    progress_bar = tqdm(dataloader, desc="Training")

    for step, batch in enumerate(progress_bar):
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)

        # Forward pass: pass `targets` to get full-sequence logits for loss
        logits, _ = model(input_ids, targets=input_ids, padding_mask=attention_mask)

        # Compute loss using -100 masking
        loss = compute_loss(logits, labels)
        loss = loss / accumulation_steps

        # Backward pass
        loss.backward()

        # Gradient accumulation
        if (step + 1) % accumulation_steps == 0:
            optimizer.step()
            if scheduler is not None:
                scheduler.step()
            optimizer.zero_grad()

        total_loss += loss.item() * accumulation_steps
        num_batches += 1

        # Update progress bar
        progress_bar.set_postfix({'loss': f'{loss.item() * accumulation_steps:.4f}'})

    return total_loss / num_batches


def evaluate(model, dataloader, device):
    """Evaluate model on validation set."""
    model.eval()
    total_loss = 0
    num_batches = 0

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating"):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            # Forward pass: pass `targets` to get full-sequence logits for loss
            logits, _ = model(input_ids, targets=input_ids, padding_mask=attention_mask)

            # Compute loss
            loss = compute_loss(logits, labels)

            total_loss += loss.item()
            num_batches += 1

    return total_loss / num_batches


def train(args):
    """Main training function."""
    if args.use_gpu:
        if not torch.cuda.is_available():
            raise RuntimeError("--use_gpu specified but CUDA is not available.")
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
    print(f"Using device: {device}")

    # Load pre-trained model first (to get max_seq_len)
    print(f"Loading pre-trained model from {args.pretrained_model_path}...")
    model = load_pretrained(args.pretrained_model_path)

    # Determine effective max length: use user-provided if >0, else cap at model limit
    effective_max_len = None
    if args.max_length and args.max_length > 0:
        effective_max_len = min(args.max_length, model.config.max_seq_len)
    else:
        effective_max_len = model.config.max_seq_len

    # Load tokenizer (apply same cap for safety)
    tokenizer = Tokenizer(max_len=effective_max_len)

    # Create dataloaders
    print("Loading Dolly-15k dataset...")
    train_loader, val_loader = create_dolly_dataloaders(
        tokenizer=tokenizer,
        batch_size=args.batch_size,
        max_length=effective_max_len,
    )

    # Enable causal mask for generation
    model.config.use_causal_mask = True

    # Train all parameters (no LoRA)
    optimizer_params = model.parameters()

    model = model.to(device)

    # Create optimizer
    optimizer = AdamW(optimizer_params, lr=args.lr, weight_decay=args.weight_decay)

    # LR scheduler (default: cosine with warmup)
    scheduler = None
    if args.scheduler == 'cosine_warmup':
        steps_per_epoch = max(1, len(train_loader) // args.gradient_accumulation_steps)
        total_steps = steps_per_epoch * args.epochs
        warmup_steps = int(total_steps * args.warmup_ratio)

        def lr_lambda(current_step: int):
            if total_steps <= 0:
                return 1.0
            # Linear warmup
            if warmup_steps > 0 and current_step < warmup_steps:
                return float(current_step) / float(max(1, warmup_steps))
            # Cosine decay to 0
            progress = float(current_step - warmup_steps) / float(max(1, total_steps - warmup_steps))
            progress = min(max(progress, 0.0), 1.0)
            return 0.5 * (1.0 + math.cos(math.pi * progress))

        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    # Training loop
    best_val_loss = float('inf')
    for epoch in range(args.epochs):
        print(f"\nEpoch {epoch + 1}/{args.epochs}")

        # Train
        train_loss = train_epoch(
            model, train_loader, optimizer, device,
            accumulation_steps=args.gradient_accumulation_steps,
            scheduler=scheduler,
        )

        # Evaluate
        val_loss = evaluate(model, val_loader, device)

        print(f"Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")

        # Save checkpoint if improved
        if val_loss < best_val_loss:
            best_val_loss = val_loss

            # Save model
            # Pack minimal model_args so this checkpoint can be loaded via load_pretrained
            model_args = {
                'vocab_size': model.config.vocab_size,
                'dim': model.config.dim,
                'dropout': model.config.dropout,
                'n_layers': model.config.n_layers,
                'n_heads': model.config.n_heads,
                'n_kv_heads': model.config.n_kv_heads,
                'max_seq_len': model.config.max_seq_len,
                'layer_norm_eps': model.config.layer_norm_eps,
                'multiple_of': model.config.multiple_of,
                'hidden_dim': model.config.hidden_dim,
                'position_embedding_type': getattr(model.config, 'position_embedding_type', 'rotary'),
                'use_cache': getattr(model.config, 'use_cache', True),
                'use_causal_mask': getattr(model.config, 'use_causal_mask', False),
            }

            checkpoint = {
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'epoch': epoch,
                'train_loss': train_loss,
                'val_loss': val_loss,
                'config': model.config,
                'model_args': model_args,
                'args': vars(args)
            }

            checkpoint_path = args.checkpoint_path or f"instruction_tuned_epoch{epoch + 1}.pt"
            torch.save(checkpoint, checkpoint_path)
            print(f"Saved checkpoint to {checkpoint_path}")

    print("\nTraining completed!")
    return model


def chat(args):
    """Interactive chat mode."""
    if args.use_gpu:
        if not torch.cuda.is_available():
            raise RuntimeError("--use_gpu specified but CUDA is not available.")
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
    print(f"Using device: {device}")

    # Load tokenizer (no truncation if max_length<=0)
    tok_max_len = args.max_length if (args.max_length and args.max_length > 0) else None
    tokenizer = Tokenizer(max_len=tok_max_len)

    # Load model (all checkpoints now include model_args)
    load_path = args.checkpoint_path if args.checkpoint_path else args.pretrained_model_path
    if not load_path:
        raise ValueError("Provide either --checkpoint_path or --pretrained_model_path")
    print(f"Loading model from {load_path}...")
    model = load_pretrained(load_path)

    # Enable causal mask for generation
    model.config.use_causal_mask = True
    # Provide EOS id to model for natural stopping
    try:
        model.params.eos_token_id = tokenizer.eos_id
    except Exception:
        if hasattr(model, 'config'):
            setattr(model.config, 'eos_token_id', tokenizer.eos_id)
    model = model.to(device)
    model.eval()

    print("\n" + "="*50)
    print("Instruction-Following Chat Interface")
    print("="*50)
    print("Type 'quit' or 'exit' to end the session")
    print("Note: Each query is independent (no conversation history)")
    print("="*50 + "\n")

    while True:
        # Get user input
        instruction = input("\n📝 Enter instruction: ").strip()

        if instruction.lower() in ['quit', 'exit']:
            print("Goodbye!")
            break

        if not instruction:
            continue

        # Format prompt
        prompt = PROMPT_TEMPLATE_NO_INPUT.format(instruction=instruction)

        # Tokenize
        input_ids = tokenizer.encode(prompt, bos=True, eos=False)
        input_tensor = torch.tensor(input_ids, dtype=torch.long, device=device).unsqueeze(0)

        # Generate response
        print("\n🤖 Response: ", end="", flush=True)

        with torch.no_grad():
            # Choose decoder based on CLI args
            if args.decoder == "generate":
                generated_ids = model.generate(
                    input_tensor,
                    max_new_tokens=args.max_new_tokens,
                    temperature=args.temperature,
                    epsilon=args.epsilon,
                )
            elif args.decoder == "sample":
                generated_ids = model.decode(
                    input_tensor,
                    max_new_tokens=args.max_new_tokens,
                    temperature=args.temperature,
                    top_p=args.top_p,
                    top_k=args.top_k,
                    repetition_penalty=args.repetition_penalty,
                )
            elif args.decoder == "beam":
                generated_ids = model.beam_search(
                    input_tensor,
                    max_new_tokens=args.max_new_tokens,
                    num_beams=args.num_beams,
                    length_penalty=args.length_penalty,
                    repetition_penalty=args.repetition_penalty,
                )
            else:
                raise ValueError(f"Unknown decoder: {args.decoder}")

            # Decode only the generated part
            generated_text = tokenizer.decode(generated_ids[0].tolist())

            # Extract only the response part
            if "### Response:" in generated_text:
                response = generated_text.split("### Response:")[-1].strip()
                print(response)
            else:
                print(generated_text[len(prompt):].strip())


def main():
    parser = argparse.ArgumentParser(description="Instruction tuning for Llama model")

    # Mode
    parser.add_argument("--mode", type=str, choices=["train", "chat"], required=True,
                        help="Mode: train or chat")

    # Model paths
    parser.add_argument("--pretrained_model_path", type=str, default="instruction_tuned.pt",
                        help="Path to pre-trained model checkpoint (default: instruction_tuned.pt; required for training, optional for chat if using checkpoint)")
    parser.add_argument("--checkpoint_path", type=str, default=None,
                        help="Path to save/load fine-tuned checkpoint")

    # Data
    parser.add_argument("--max_length", type=int, default=0,
                        help="Max sequence length (0=auto cap at model limit)")

    # Training hyperparameters
    parser.add_argument("--batch_size", type=int, default=4,
                        help="Batch size for training")
    parser.add_argument("--epochs", type=int, default=3,
                        help="Number of training epochs")
    parser.add_argument("--lr", type=float, default=1e-5,
                        help="Learning rate (default 1e-5; typical 5e-6~1e-5)")
    parser.add_argument("--weight_decay", type=float, default=0.01,
                        help="Weight decay for AdamW")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=4,
                        help="Number of gradient accumulation steps")
    parser.add_argument("--scheduler", type=str, default="cosine_warmup", choices=["none", "cosine_warmup"],
                        help="LR scheduler to use (default: cosine_warmup)")
    parser.add_argument("--warmup_ratio", type=float, default=0.04,
                        help="Warmup ratio for cosine_warmup scheduler (default 0.04; typical 0.03~0.05)")

    # (LoRA removed)

    # Generation parameters (for chat mode)
    parser.add_argument("--max_new_tokens", type=int, default=256,
                        help="Maximum number of tokens to generate")
    parser.add_argument("--decoder", type=str, default="generate",
                        choices=["generate", "sample", "beam"],
                        help="Decoder: 'generate' (epsilon), 'sample' (temp/top-p/top-k/rep), 'beam' (beam search)")
    # Legacy generate (epsilon sampling)
    parser.add_argument("--epsilon", type=float, default=0.05,
                        help="Epsilon threshold for legacy generate()")
    # Sampling decoder parameters
    parser.add_argument("--temperature", type=float, default=0.7,
                        help="Temperature for sampling/decoding; <=0 for greedy")
    parser.add_argument("--top_p", type=float, default=1.0,
                        help="Top-p (nucleus) threshold for 'sample' decoder")
    parser.add_argument("--top_k", type=int, default=0,
                        help="Top-k for 'sample' decoder (0 disables)")
    parser.add_argument("--repetition_penalty", type=float, default=1.0,
                        help=">1.0 penalizes repeated tokens for 'sample'/'beam'")
    # Beam search parameters
    parser.add_argument("--num_beams", type=int, default=4,
                        help="Number of beams for 'beam' decoder")
    parser.add_argument("--length_penalty", type=float, default=1.0,
                        help="Length penalty for 'beam' decoder (GNMT style)")
    # Early stopping is always EOS-based; no flag

    # Other
    parser.add_argument("--use_gpu", action="store_true",
                        help="Use GPU if available")
    # (Debug mode removed)
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed")

    args = parser.parse_args()

    # Set random seed
    torch.manual_seed(args.seed)
    if args.use_gpu and torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)

    # Run selected mode
    if args.mode == "train":
        train(args)
    else:
        chat(args)


if __name__ == "__main__":
    main()
