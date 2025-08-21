import os
import argparse
from tqdm import tqdm
import torch
from apps.data import FolderDataset
from apps.preprocess import AddHeuristicFillIn

def main(args):
    """
    Applies the AddHeuristicFillIn transform to a raw dataset and saves
    the processed graphs to a new directory.
    """
    # Path to the original (raw) data
    raw_path = os.path.join(args.dataset_path, "raw")
    # Path where the new, processed graphs will be saved
    processed_path = os.path.join(args.dataset_path, "processed")

    # Create the transform we want to apply
    transform = AddHeuristicFillIn(K=args.k)
    print(f"Applying transform: {transform}")

    for split in ["train", "test"]:
        raw_split_path = os.path.join(args.dataset_path, split)
        processed_split_path = os.path.join(processed_path, split)
        os.makedirs(processed_split_path, exist_ok=True)

        print(f"\nProcessing '{split}' split...")
        # Load the raw dataset
        raw_dataset = FolderDataset(folder_path=raw_split_path)

        for i in tqdm(range(len(raw_dataset)), desc=f"  -> Preprocessing {split}"):
            raw_data = raw_dataset.get(i)
            # Apply the expensive transform
            processed_data = transform(raw_data)
            # Save the new, augmented graph
            save_path = os.path.join(processed_split_path, raw_dataset.files[i])
            torch.save(processed_data, save_path)

    print("\nPreprocessing complete.")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_path", type=str, required=True, help="Path to the root of the dataset.")
    parser.add_argument("--k", type=int, default=5, help="Number of fill-in edges to add.")
    args = parser.parse_args()
    main(args)