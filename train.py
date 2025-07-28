import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from torch.optim.lr_scheduler import ReduceLROnPlateau
from tqdm import tqdm
import argparse
import os
from torch.amp import autocast, GradScaler

from model import Tokenizer, Image2LatexModel
from data_loader import LatexDataset, collate_fn, tokenize_latex # Assuming tokenize_latex is in data_loader

def train_one_epoch(model, data_loader, criterion, optimizer, device, epoch, total_epochs, scaler):
    """
    Performs one full training pass over the dataset using mixed precision.
    """
    model.train()  # Set model to training mode
    
    total_loss = 0.0
    progress_bar = tqdm(data_loader, desc=f"Epoch {epoch+1}/{total_epochs} [Training]")
    
    for images, labels in progress_bar:
        images = images.to(device)
        labels = labels.to(device)
        
        # For teacher forcing, the target sequence is the label sequence shifted by one.
        # The input to the decoder is the sequence without the last token (<eos>).
        # The target for the loss function is the sequence without the first token (<sos>).
        tgt_in = labels[:, :-1]
        tgt_out = labels[:, 1:]
        
        # Create a padding mask for the input sequence
        # 1s for padding tokens, 0s for real tokens
        tgt_padding_mask = (tgt_in == 0) # Assumes pad_id is 0

        # Zero the gradients
        optimizer.zero_grad(set_to_none=True) # More efficient
        
        # Use autocast for mixed precision forward pass
        with autocast(device_type=device.type, dtype=torch.float16):
            predictions = model(images, tgt_in, tgt_padding_mask=tgt_padding_mask)
            loss = criterion(predictions.view(-1, model.decoder.generator.out_features), tgt_out.reshape(-1))
        
        # Scale loss and perform backward pass
        scaler.scale(loss).backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0) # Clip gradients
        scaler.step(optimizer)
        scaler.update()
        
        total_loss += loss.item()
        progress_bar.set_postfix(loss=loss.item())
        
    return total_loss / len(data_loader)

def validate(model, data_loader, criterion, device):
    """
    Evaluates the model on the validation set.
    """
    model.eval()  # Set model to evaluation mode
    total_loss = 0.0
    
    with torch.no_grad():
        progress_bar = tqdm(data_loader, desc="Validating")
        for images, labels in progress_bar:
            images = images.to(device)
            labels = labels.to(device)
            
            tgt_in = labels[:, :-1]
            tgt_out = labels[:, 1:]
            tgt_padding_mask = (tgt_in == 0)
            
            # Use autocast for validation as well for consistency
            with autocast(device_type=device.type, dtype=torch.float16):
                predictions = model(images, tgt_in, tgt_padding_mask=tgt_padding_mask)
                loss = criterion(predictions.view(-1, model.decoder.generator.out_features), tgt_out.reshape(-1))
            total_loss += loss.item()
            progress_bar.set_postfix(loss=loss.item())

    return total_loss / len(data_loader)


def main(args):
    # --- Setup ---
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # --- Tokenizer ---
    tokenizer = Tokenizer(args.vocab_path)
    vocab_size = tokenizer.vocab_size
    pad_id = tokenizer.pad_id
    print(f"Tokenizer loaded. Vocabulary size: {vocab_size}")
    
    # --- Datasets and DataLoaders ---
    print("Loading data...")
    full_train_dataset = LatexDataset(data_dir=args.train_data_dir, tokenizer=tokenizer, augment=True)
    
    # Use a fraction of the dataset if specified for faster training
    if args.dataset_frac < 1.0:
        dataset_size = len(full_train_dataset)
        subset_size = int(dataset_size * args.dataset_frac)
        indices = torch.randperm(dataset_size).tolist()[:subset_size]
        train_dataset = Subset(full_train_dataset, indices)
        print(f"Using a SUBSET of the training data: {subset_size}/{dataset_size} samples ({args.dataset_frac*100:.1f}%)")
    else:
        train_dataset = full_train_dataset
        print(f"Using the FULL training dataset: {len(train_dataset)} samples.")

    val_dataset = LatexDataset(data_dir=args.val_data_dir, tokenizer=tokenizer, augment=False)
    
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn, num_workers=args.num_workers, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn, num_workers=args.num_workers, pin_memory=True)
    print("Data loaded.")
    
    # --- Model, Loss, Optimizer ---
    print("Initializing model...")
    model = Image2LatexModel(vocab_size=vocab_size, d_model=args.d_model, nhead=args.nhead, num_decoder_layers=args.decoder_layers).to(device)
    
    # Loss function that ignores padding tokens
    criterion = nn.CrossEntropyLoss(ignore_index=pad_id)
    
    # Optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    
    # Initialize GradScaler for mixed precision
    scaler = GradScaler(enabled=args.amp)
    
    # --- Load Checkpoint ---
    start_epoch = 0
    best_val_loss = float('inf')
    if os.path.exists(args.checkpoint_path):
        print(f"Resuming training from checkpoint: {args.checkpoint_path}")
        checkpoint = torch.load(args.checkpoint_path)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        best_val_loss = checkpoint.get('best_val_loss', float('inf'))
        print(f"Resuming from epoch {start_epoch}. Best validation loss so far: {best_val_loss:.4f}")

    # --- Training Loop ---
    print("Starting training with Mixed Precision...")
    for epoch in range(start_epoch, args.epochs):
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device, epoch, args.epochs, scaler)
        val_loss = validate(model, val_loader, criterion, device)
        
        # Update learning rate
        scheduler = ReduceLROnPlateau(optimizer, 'min', patience=args.lr_patience, factor=0.5)
        scheduler.step(val_loss)
        
        print(f"\nEpoch {epoch+1}/{args.epochs} -> Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, LR: {optimizer.param_groups[0]['lr']}\n")
        
        is_best = val_loss < best_val_loss
        if is_best:
            best_val_loss = val_loss
            print(f"New best model found! Saving to {args.save_path}")
            torch.save(model.state_dict(), args.save_path)

        # Save checkpoint
        print("Saving checkpoint...")
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'best_val_loss': best_val_loss,
        }, args.checkpoint_path)

    print("Training finished.")
    print(f"Best validation loss: {best_val_loss:.4f}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Train the Image-to-LaTeX model with Mixed Precision and Checkpointing.")
    
    # Paths
    parser.add_argument('--vocab_path', type=str, default='vocab.json', help='Path to the vocabulary file.')
    parser.add_argument('--train_data_dir', type=str, default='mathwriting-2024/train', help='Directory for training data.')
    parser.add_argument('--val_data_dir', type=str, default='mathwriting-2024/valid', help='Directory for validation data.')
    parser.add_argument('--save_path', type=str, default='best_model_full.pth', help='Path to save the best performing model.')
    parser.add_argument('--checkpoint_path', type=str, default='checkpoint.pth', help='Path to save the training checkpoint.')

    # Model hyperparameters
    parser.add_argument('--d_model', type=int, default=512, help='Dimension of the model.')
    parser.add_argument('--decoder_layers', type=int, default=6, help='Number of transformer decoder layers.')
    parser.add_argument('--nhead', type=int, default=8, help='Number of heads in the multiheadattention models.')

    # Training hyperparameters
    parser.add_argument('--epochs', type=int, default=20, help='Number of training epochs.')
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size for training. Adjust based on your VRAM.')
    parser.add_argument('--lr', type=float, default=5e-5, help='Learning rate.')
    parser.add_argument('--lr_patience', type=int, default=3, help='Patience for learning rate scheduler (in epochs).')
    parser.add_argument('--num_workers', type=int, default=4, help='Number of workers for DataLoader.')
    parser.add_argument('--dataset_frac', type=float, default=0.2, help='Fraction of the training dataset to use (e.g., 0.1 for 10%). Default is 20%.')
    
    # AMP argument
    parser.add_argument('--amp', action='store_true', default=True, help='Enable Automatic Mixed Precision training.')

    args = parser.parse_args()
    main(args) 