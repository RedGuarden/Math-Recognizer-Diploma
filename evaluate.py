import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import argparse
import editdistance # For calculating Character Error Rate

# Dynamic model import
from data_loader import LatexDataset
from recognize import beam_search_decode # Re-use the beam search from recognize.py

def evaluate_model(model, data_loader, tokenizer, device, beam_width):
    """
    Evaluates the model on a given dataset and calculates metrics.
    """
    model.eval()
    
    total_samples = 0
    correct_predictions = 0
    total_cer = 0.0
    
    # Let's re-iterate directly over the dataset to get ground truth easily
    for i in tqdm(range(len(data_loader.dataset)), desc="Evaluating Samples"):
        image_tensor, label_tensor = data_loader.dataset[i]
        
        # Get ground truth LaTeX string (by decoding the label tensor)
        ground_truth_latex = tokenizer.decode(label_tensor.tolist(), strip_special_tokens=True)
        
        # Generate prediction
        predicted_latex = beam_search_decode(model, image_tensor, tokenizer, device, beam_width=beam_width, max_len=256)
        
        # Check for perfect match
        if predicted_latex == ground_truth_latex:
            correct_predictions += 1
            
        # Calculate Character Error Rate (CER)
        # We use editdistance, which is Levenshtein distance
        cer = editdistance.eval(predicted_latex, ground_truth_latex) / len(ground_truth_latex) if len(ground_truth_latex) > 0 else 0
        total_cer += cer
        
        total_samples += 1
        
    # Calculate final metrics
    expression_recognition_rate = (correct_predictions / total_samples) * 100
    average_cer = (total_cer / total_samples) * 100
    
    return expression_recognition_rate, average_cer


def main(args):
    # --- Setup ---
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # --- Dynamic Model and Tokenizer Loading ---
    if args.model_version == 'v1':
        from model import Tokenizer, Image2LatexModel
    elif args.model_version == 'v2':
        from model_v2 import Tokenizer, Image2LatexModelV2 as Image2LatexModel
    else:
        raise ValueError("Invalid model version specified. Choose 'v1' or 'v2'.")

    # --- Load Tokenizer ---
    tokenizer = Tokenizer(args.vocab_path)
    print(f"Tokenizer loaded. Using model version: {args.model_version}")

    # --- Load Model ---
    print("Loading model...")
    model = Image2LatexModel(vocab_size=tokenizer.vocab_size, d_model=args.d_model, nhead=args.nhead, num_decoder_layers=args.decoder_layers)
    model.load_state_dict(torch.load(args.model_path, map_location=device))
    model.to(device)
    print("Model loaded.")
    
    # --- Test Dataset ---
    test_dataset = LatexDataset(data_dir=args.test_data_dir, tokenizer=tokenizer, augment=False)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)
    
    # --- Run Evaluation ---
    err, cer = evaluate_model(model, test_loader, tokenizer, device, beam_width=args.beam_width)
    
    print("\n--- Evaluation Complete ---")
    print(f"  Total Test Samples: {len(test_dataset)}")
    print(f"  Expression Recognition Rate (ERR): {err:.2f}%")
    print(f"  Character Error Rate (CER):        {cer:.2f}%")
    print("---------------------------\n")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Evaluate the Image-to-LaTeX model on the test set.")
    
    # Paths
    parser.add_argument('--vocab_path', type=str, default='vocab.json', help='Path to the vocabulary file.')
    parser.add_argument('--model_path', type=str, default='best_model_full.pth', help='Path to the trained model file.')
    parser.add_argument('--test_data_dir', type=str, default='mathwriting-2024/test', help='Directory for test data.')

    # Model Version
    parser.add_argument('--model_version', type=str, default='v1', choices=['v1', 'v2'], help='Specify which model version to use (v1: ResNet, v2: EfficientNet).')

    # Model hyperparameters (must match the trained model)
    parser.add_argument('--d_model', type=int, default=512, help='Dimension of the model.')
    parser.add_argument('--decoder_layers', type=int, default=6, help='Number of transformer decoder layers.')
    parser.add_argument('--nhead', type=int, default=8, help='Number of heads in the multiheadattention models.')
    
    # Evaluation hyperparameters
    parser.add_argument('--beam_width', type=int, default=5, help='Beam width for beam search decoding.')

    args = parser.parse_args()
    main(args) 