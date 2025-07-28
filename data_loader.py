import os
import xml.etree.ElementTree as ET
import matplotlib.pyplot as plt
import random
import numpy as np
import cv2
import re
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from model import Tokenizer

def parse_inkml(inkml_file):
    """
    Parses an InkML file and extracts stroke data.

    Args:
        inkml_file (str): Path to the InkML file.

    Returns:
        tuple: A tuple containing:
            - strokes (list of lists of tuples): A list of strokes, where each stroke
              is a list of (x, y) coordinates.
            - latex_label (str): The normalized LaTeX label for the expression.
    """
    tree = ET.parse(inkml_file)
    root = tree.getroot()
    
    # Namespace dictionary to handle the InkML namespace
    ns = {'inkml': 'http://www.w3.org/2003/InkML'}
    
    # Extract strokes
    strokes = []
    for trace in root.findall('inkml:trace', ns):
        points = trace.text.strip().split(',')
        stroke = []
        for point in points:
            coords = point.strip().split(' ')
            if len(coords) >= 2:
                try:
                    stroke.append((float(coords[0]), float(coords[1])))
                except ValueError:
                    # Handle potential malformed coordinate pairs
                    pass
        if stroke:
            strokes.append(stroke)
            
    # Extract normalized LaTeX label
    latex_label = ''
    for annotation in root.findall('inkml:annotation', ns):
        if annotation.get('type') == 'normalizedLabel':
            # In the excerpt, the label is in 'normalizedLabel', but in the full dataset it might be under a different tag.
            # Let's also check for a 'truth' annotation which is common.
            latex_label = annotation.text.strip()
            if latex_label:
                break
    
    if not latex_label:
        # Fallback for other possible annotation types
        for annotation in root.findall('inkml:annotation', ns):
            if annotation.get('type') == 'truth':
                latex_label = annotation.text.strip()
                if latex_label:
                    break
            
    return strokes, latex_label

def visualize_strokes(strokes, latex_label, output_path='visualization.png'):
    """
    Visualizes strokes from an InkML file and saves it as an image.

    Args:
        strokes (list): The list of strokes to visualize.
        latex_label (str): The LaTeX label to use as the title.
        output_path (str): The path to save the output image.
    """
    plt.figure(figsize=(10, 5))
    for stroke in strokes:
        # Separate x and y coordinates
        x_coords, y_coords = zip(*stroke)
        # Invert y-axis to match typical image coordinates (origin at top-left)
        plt.plot(x_coords, [-y for y in y_coords], marker='o', linestyle='-')

    plt.title(f'Rendered InkML\nLaTeX: ${latex_label}$')
    plt.xlabel('X coordinate')
    plt.ylabel('Y coordinate (inverted)')
    plt.gca().set_aspect('equal', adjustable='box')
    plt.grid(True)
    
    # Save the plot to a file
    plt.savefig(output_path)
    print(f"Visualization saved to {output_path}")
    plt.close()

def strokes_to_image(strokes, image_size=(224, 224), padding=16, bg_color=0, stroke_color=255, thickness=1):
    """
    Renders strokes into a fixed-size numpy array image.

    Args:
        strokes (list): List of strokes (each a list of (x, y) points).
        image_size (tuple): The (width, height) of the output image.
        padding (int): Padding around the drawing.
        bg_color (int): Background color (0-255).
        stroke_color (int): Stroke color (0-255).
        thickness (int): Thickness of the strokes.

    Returns:
        np.ndarray: A grayscale image as a numpy array.
    """
    if not strokes:
        return np.full((image_size[1], image_size[0]), bg_color, dtype=np.uint8)

    # Find bounding box of the entire drawing
    all_points = [point for stroke in strokes for point in stroke]
    min_x = min(p[0] for p in all_points)
    max_x = max(p[0] for p in all_points)
    min_y = min(p[1] for p in all_points)
    max_y = max(p[1] for p in all_points)

    # Drawing dimensions
    drawing_width = max_x - min_x
    drawing_height = max_y - min_y

    if drawing_width == 0 and drawing_height == 0:
        return np.full((image_size[1], image_size[0]), bg_color, dtype=np.uint8)

    # Target drawing area within the image
    canvas_width = image_size[0] - 2 * padding
    canvas_height = image_size[1] - 2 * padding

    # Calculate scaling factor, preserving aspect ratio
    scale = 1.0
    if drawing_width > 0:
        scale = min(scale, canvas_width / drawing_width)
    if drawing_height > 0:
        scale = min(scale, canvas_height / drawing_height)

    # New dimensions after scaling
    new_width = int(drawing_width * scale)
    new_height = int(drawing_height * scale)

    # Offset to center the drawing
    offset_x = padding + (canvas_width - new_width) // 2
    offset_y = padding + (canvas_height - new_height) // 2

    # Create blank image
    image = np.full((image_size[1], image_size[0]), bg_color, dtype=np.uint8)

    # Draw strokes
    for stroke in strokes:
        # Scale and shift points
        points_scaled = []
        for x, y in stroke:
            new_x = int((x - min_x) * scale) + offset_x
            new_y = int((y - min_y) * scale) + offset_y
            points_scaled.append((new_x, new_y))

        # Draw lines between consecutive points
        for i in range(len(points_scaled) - 1):
            p1 = points_scaled[i]
            p2 = points_scaled[i+1]
            cv2.line(image, p1, p2, stroke_color, thickness, cv2.LINE_AA)

    return image

def tokenize_latex(latex_string):
    """
    Tokenizes a LaTeX string into a list of tokens.
    - Keeps LaTeX commands (e.g., '\\frac', '\\sin') as single tokens.
    - Treats special characters '{', '}', '^', '_' as individual tokens.
    - Splits all other characters.
    """
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

class LatexDataset(Dataset):
    """
    PyTorch Dataset for the InkML data.
    """
    def __init__(self, data_dir, tokenizer, max_len=256, augment=False):
        self.data_dir = data_dir
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.augment = augment
        
        self.inkml_files = [os.path.join(data_dir, f) for f in os.listdir(data_dir) if f.endswith('.inkml')]
        
        # Transformations
        if self.augment:
            # Augmentation pipeline for training
            self.transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.RandomAffine(degrees=5, translate=(0.05, 0.05), scale=(0.95, 1.05), shear=5),
                transforms.Normalize(mean=[0.5], std=[0.5])
            ])
        else:
            # Standard pipeline for validation/testing
            self.transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5], std=[0.5])
            ])

    def __len__(self):
        return len(self.inkml_files)

    def __getitem__(self, idx):
        file_path = self.inkml_files[idx]
        
        # 1. Parse InkML to get strokes and label
        strokes, latex_label = parse_inkml(file_path)
        
        # 2. Render strokes to image
        # Note: strokes_to_image returns a (H, W) numpy array
        image = strokes_to_image(strokes)
        
        # 3. Apply transformations to the image
        # ToTensor expects (H, W, C), so we need to add a channel dimension
        image = self.transform(image)
        
        # 4. Tokenize the LaTeX label
        latex_tokens = tokenize_latex(latex_label)
        
        # 5. Encode tokens and add special <sos> and <eos> tokens
        encoded_label = [self.tokenizer.sos_id] + self.tokenizer.encode(latex_tokens) + [self.tokenizer.eos_id]
        
        # Truncate if longer than max_len
        if len(encoded_label) > self.max_len:
            encoded_label = encoded_label[:self.max_len-1] + [self.tokenizer.eos_id]
            
        return image, torch.LongTensor(encoded_label)


def collate_fn(batch):
    """
    Custom collate function to handle padding in batches.
    """
    # Unzip the batch
    images, labels = zip(*batch)
    
    # Stack images into a single tensor
    images = torch.stack(images, 0)
    
    # Pad labels to the length of the longest sequence in the batch
    # We need to know the padding index from the first sample's tokenizer
    pad_id = labels[0][-1] # A bit of a hack, assuming tokenizer is the same
    if isinstance(batch[0][1], torch.Tensor):
        pad_id = batch[0][1].new_tensor(0).squeeze().item() # Safest way to get pad_id if we assume tokenizer is consistent
        # A better way would be to pass tokenizer to collate_fn, but this is simpler
        
    # Let's find the pad_id more reliably
    # This is a bit of a workaround. In a full app, the tokenizer would be more accessible.
    # For now, we assume 0 is <pad>.
    pad_id = 0
    
    padded_labels = torch.nn.utils.rnn.pad_sequence(
        labels, batch_first=True, padding_value=pad_id
    )
    
    return images, padded_labels


if __name__ == '__main__':
    print("Testing Dataset and DataLoader...")
    
    # 1. Initialize Tokenizer
    try:
        tokenizer = Tokenizer('vocab.json')
        print(f"Tokenizer loaded. Vocab size: {tokenizer.vocab_size}")
    except Exception as e:
        print(f"Could not load tokenizer: {e}. Make sure vocab.json is present.")
        exit()

    # 2. Initialize Dataset
    dataset_dir = 'mathwriting-2024-excerpt/train'
    try:
        dataset = LatexDataset(data_dir=dataset_dir, tokenizer=tokenizer)
        print(f"Dataset created successfully. Found {len(dataset)} samples.")
    except Exception as e:
        print(f"Error creating dataset: {e}")
        exit()
        
    # 3. Initialize DataLoader
    # Note: A large dataset might be slow to iterate through for a test.
    # We'll just check if we can get one batch.
    
    # Let's create a smaller subset for testing purposes
    subset_indices = list(range(100)) # Use first 100 samples for test
    subset = torch.utils.data.Subset(dataset, subset_indices)

    BATCH_SIZE = 8
    
    data_loader = DataLoader(
        subset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        collate_fn=collate_fn
    )
    
    print(f"DataLoader created with batch size {BATCH_SIZE}.")
    
    # 4. Fetch one batch and check shapes
    try:
        images, labels = next(iter(data_loader))
        
        print("\n--- Batch Test ---")
        print(f"Images batch shape: {images.shape}")
        print(f"Labels batch shape: {labels.shape}")
        
        # Verification
        assert images.shape[0] == BATCH_SIZE
        assert images.shape[1] == 1 # Grayscale channel
        assert images.shape[2] == 224 and images.shape[3] == 224
        assert labels.shape[0] == BATCH_SIZE
        
        print("\nSuccessfully fetched a batch. Shapes are correct.")
        
        # Optional: decode one label from the batch to check
        first_label_decoded = tokenizer.decode(labels[0].tolist())
        print(f"Decoded first label in batch: {first_label_decoded}")
        
    except Exception as e:
        print(f"\nAn error occurred while fetching a batch: {e}") 