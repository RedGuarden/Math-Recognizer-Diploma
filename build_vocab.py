import os
import json
import re
from collections import Counter
from tqdm import tqdm
import argparse

# Import the parser from our data loader
from data_loader import parse_inkml

def tokenize_latex(latex_string):
    """
    Tokenizes a LaTeX string into a list of tokens.
    - Keeps LaTeX commands (e.g., '\\frac', '\\sin') as single tokens.
    - Treats special characters '{', '}', '^', '_' as individual tokens.
    - Splits all other characters.
    """
    # This regex finds LaTeX commands, special symbols, or any single character.
    token_regex = r'(\\([a-zA-Z]+|[\\%]))|([{}@#$&_])|(.)'
    tokens = []
    for match in re.finditer(token_regex, latex_string):
        if match.group(1): # LaTeX command like \frac or \%
            tokens.append(match.group(1))
        elif match.group(3): # Special character
            tokens.append(match.group(3))
        elif match.group(4): # Any other character
            tokens.append(match.group(4))
    return tokens

def build_vocab(data_dir, output_file, min_freq=1):
    """
    Builds a vocabulary from all .inkml files in the specified directory.
    
    Args:
        data_dir (str): The directory containing the .inkml files (e.g., 'mathwriting-2024-excerpt/train').
        output_file (str): Path to save the vocabulary JSON file.
        min_freq (int): The minimum frequency for a token to be included in the vocabulary.
    """
    print(f"Scanning .inkml files in '{data_dir}'...")
    
    if not os.path.isdir(data_dir):
        print(f"Error: Directory not found at '{data_dir}'")
        return

    inkml_files = [os.path.join(data_dir, f) for f in os.listdir(data_dir) if f.endswith('.inkml')]
    
    if not inkml_files:
        print(f"No .inkml files found in '{data_dir}'.")
        return

    token_counts = Counter()
    
    # Using tqdm for a progress bar
    for file_path in tqdm(inkml_files, desc="Parsing files and tokenizing"):
        _, latex_label = parse_inkml(file_path)
        if latex_label:
            tokens = tokenize_latex(latex_label)
            token_counts.update(tokens)
            
    print(f"Found {len(token_counts)} unique tokens.")

    # Filter tokens by minimum frequency
    vocab = [token for token, count in token_counts.items() if count >= min_freq]
    print(f"Keeping {len(vocab)} tokens with minimum frequency >= {min_freq}.")

    # Add special tokens
    # <pad>: Padding token
    # <sos>: Start of Sequence token
    # <eos>: End of Sequence token
    # <unk>: Unknown token (for tokens not in vocabulary)
    special_tokens = ['<pad>', '<sos>', '<eos>', '<unk>']
    
    # Combine and create the final vocabulary list
    full_vocab = special_tokens + sorted(vocab)
    
    # Create token-to-index and index-to-token mappings
    token_to_id = {token: i for i, token in enumerate(full_vocab)}
    id_to_token = {i: token for i, token in enumerate(full_vocab)}
    
    # Save to a JSON file
    vocab_data = {
        'token_to_id': token_to_id,
        'id_to_token': id_to_token
    }
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(vocab_data, f, ensure_ascii=False, indent=4)
        
    print(f"Vocabulary successfully built and saved to '{output_file}'")
    print(f"Total vocabulary size: {len(full_vocab)}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Build vocabulary from InkML dataset.")
    parser.add_argument('--data_dir', type=str, default='mathwriting-2024-excerpt/train',
                        help='Directory containing the training .inkml files.')
    parser.add_argument('--out_file', type=str, default='vocab.json',
                        help='Path to save the output vocabulary file.')
    parser.add_argument('--min_freq', type=int, default=1,
                        help='Minimum token frequency to be included in the vocabulary.')
    
    args = parser.parse_args()
    
    build_vocab(args.data_dir, args.out_file, args.min_freq) 