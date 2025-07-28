import torch
import argparse
import os
import random
import cv2
from model import Tokenizer, Image2LatexModel
from data_loader import parse_inkml, strokes_to_image
from torchvision import transforms
import heapq

def beam_search_decode(model, image_tensor, tokenizer, device, beam_width=5, max_len=150):
    """
    Generates a LaTeX sequence using Beam Search decoding.
    """
    model.eval()
    
    with torch.no_grad():
        image_tensor = image_tensor.unsqueeze(0).to(device)
        
        # Encode the image
        encoded_img = model.encoder(image_tensor)
        memory = model.encoder_projection(encoded_img) # (1, seq_len_encoder, d_model)
        
        # Start with a beam containing just the <sos> token
        # Each item in the beam is a tuple (log_probability, sequence)
        initial_beam = [(0.0, [tokenizer.sos_id])]
        
        for _ in range(max_len):
            new_beam = []
            
            for log_prob, seq in initial_beam:
                if seq[-1] == tokenizer.eos_id:
                    # This sequence is finished, keep it as is
                    heapq.heappush(new_beam, (log_prob, seq))
                    continue

                tgt_tensor = torch.LongTensor(seq).unsqueeze(0).to(device)
                tgt_mask = model.generate_square_subsequent_mask(tgt_tensor.size(1), device)
                
                output = model.decoder(memory, tgt_tensor, tgt_mask=tgt_mask)
                pred_logits = output[:, -1, :] # Get logits for the last token
                
                # Use log_softmax to get log probabilities
                pred_log_probs = torch.nn.functional.log_softmax(pred_logits, dim=-1)
                
                # Get the top N candidates
                topk_log_probs, topk_ids = torch.topk(pred_log_probs, beam_width, dim=-1)
                
                for i in range(beam_width):
                    new_seq = seq + [topk_ids[0, i].item()]
                    new_log_prob = log_prob + topk_log_probs[0, i].item()
                    heapq.heappush(new_beam, (new_log_prob, new_seq))
            
            # Prune the beam to keep only the top N candidates
            # We use a heap, so we can efficiently get the N largest log_probs
            # (log probabilities are negative, so we want the smallest absolute values)
            initial_beam = heapq.nlargest(beam_width, new_beam, key=lambda x: x[0])

            # Stop if all top candidates have finished
            if all(seq[-1] == tokenizer.eos_id for _, seq in initial_beam):
                break
        
        # The best sequence is the one with the highest log probability
        best_log_prob, best_seq = initial_beam[0]
        
        return tokenizer.decode(best_seq, strip_special_tokens=True)

def recognize_formula(model, image_tensor, tokenizer, device, max_len=100):
    """
    Generates a LaTeX sequence for a given image tensor using greedy decoding.
    """
    model.eval()  # Set model to evaluation mode
    
    with torch.no_grad():
        # Add a batch dimension and send to device
        image_tensor = image_tensor.unsqueeze(0).to(device)
        
        # Encode the image
        encoded_img = model.encoder(image_tensor)
        memory = model.encoder_projection(encoded_img) # (1, seq_len_encoder, d_model)
        
        # Start decoding with the <sos> token
        tgt_seq = [tokenizer.sos_id]
        
        for _ in range(max_len):
            # Convert current sequence to tensor
            tgt_tensor = torch.LongTensor(tgt_seq).unsqueeze(0).to(device)
            
            # Create a causal mask for the decoder
            tgt_mask = model.generate_square_subsequent_mask(tgt_tensor.size(1), device)
            
            # Get predictions
            output = model.decoder(memory, tgt_tensor, tgt_mask=tgt_mask) # (1, seq_len, vocab_size)
            
            # Get the last token's prediction (greedy search)
            pred_token_id = output.argmax(2)[:, -1].item()
            
            # Append predicted token
            tgt_seq.append(pred_token_id)
            
            # Stop if <eos> token is predicted
            if pred_token_id == tokenizer.eos_id:
                break
                
        # Decode the sequence of IDs back to a string
        predicted_latex = tokenizer.decode(tgt_seq, strip_special_tokens=True)
        return predicted_latex

def main(args):
    # --- Setup ---
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # --- Load Tokenizer ---
    tokenizer = Tokenizer(args.vocab_path)
    print("Tokenizer loaded.")

    # --- Load Model ---
    print("Loading model...")
    model = Image2LatexModel(vocab_size=tokenizer.vocab_size, d_model=args.d_model, nhead=args.nhead, num_decoder_layers=args.decoder_layers)
    model.load_state_dict(torch.load(args.model_path, map_location=device))
    model.to(device)
    print("Model loaded.")

    # --- Select a random file for recognition ---
    if not os.path.isdir(args.test_data_dir):
        print(f"Error: Test data directory not found at '{args.test_data_dir}'")
        return

    inkml_files = [f for f in os.listdir(args.test_data_dir) if f.endswith('.inkml')]
    if not inkml_files:
        print(f"No .inkml files found in '{args.test_data_dir}'")
        return
        
    random_file = random.choice(inkml_files)
    file_path = os.path.join(args.test_data_dir, random_file)
    print(f"\nSelected random file for testing: {file_path}")

    # --- Prepare the image ---
    strokes, ground_truth_latex = parse_inkml(file_path)
    image_numpy = strokes_to_image(strokes)
    
    # Save the input image for inspection
    input_image_path = "recognition_input.png"
    cv2.imwrite(input_image_path, 255 - image_numpy)
    print(f"Input image saved to '{input_image_path}'")
    
    # Apply transformations
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5])
    ])
    image_tensor = transform(image_numpy)
    
    # --- Perform Recognition ---
    print("\n--- Performing Greedy Search ---")
    greedy_latex = recognize_formula(model, image_tensor, tokenizer, device)
    print(f"  Predicted (Greedy): {greedy_latex}")

    print("\n--- Performing Beam Search ---")
    beam_latex = beam_search_decode(model, image_tensor, tokenizer, device, beam_width=args.beam_width)
    
    # --- Display Results ---
    print("\n\n--- Recognition Result ---")
    print(f"  Ground Truth:         {ground_truth_latex}")
    print(f"  Predicted (Beam Search): {beam_latex}")
    

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Recognize a handwritten formula from an InkML file.")
    
    # Paths
    parser.add_argument('--vocab_path', type=str, default='vocab.json', help='Path to the vocabulary file.')
    parser.add_argument('--model_path', type=str, default='best_model_full.pth', help='Path to the trained model file.')
    parser.add_argument('--test_data_dir', type=str, default='mathwriting-2024/test', help='Directory for test data.')

    # Model hyperparameters (should match the trained model)
    parser.add_argument('--d_model', type=int, default=512, help='Dimension of the model.')
    parser.add_argument('--decoder_layers', type=int, default=6, help='Number of transformer decoder layers.')
    parser.add_argument('--nhead', type=int, default=8, help='Number of heads in the multiheadattention models.')

    # Recognition hyperparameters
    parser.add_argument('--beam_width', type=int, default=5, help='Beam width for beam search decoding.')

    args = parser.parse_args()
    main(args) 