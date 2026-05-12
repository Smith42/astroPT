import argparse
import os
import glob
import pyarrow as pa
import pyarrow.ipc as ipc
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter
from tqdm import tqdm

def analyze_token_distribution(data_dir, output_dir, max_files=None):
    os.makedirs(output_dir, exist_ok=True)
    
    if os.path.isfile(data_dir):
        arrow_files = [data_dir]
    else:
        arrow_files = glob.glob(os.path.join(data_dir, "**", "*.arrow"), recursive=True)
        
    if not arrow_files:
        print(f"No .arrow files found at {data_dir}")
        return
        
    print(f"Found {len(arrow_files)} .arrow files.")
    if max_files is not None and not os.path.isfile(data_dir):
        arrow_files = arrow_files[:max_files]
        print(f"Limiting to {max_files} files for analysis.")
        
    token_counts = {"spectra_tokens": Counter(), "image_tokens": Counter()}
    # Debug: show schema names from the first file found
    try:
        with ipc.open_stream(arrow_files[0]) as reader:
            print(f"Schema names: {reader.schema.names}")
    except Exception:
        # Fallback for File format just in case
        try:
            with ipc.open_file(arrow_files[0]) as reader:
                print(f"Schema names (File format): {reader.schema.names}")
        except:
            pass
    
    for file_path in tqdm(arrow_files, desc="Analyzing files"):
        try:
            try:
                with ipc.open_stream(file_path) as reader:
                    table = reader.read_all()
            except Exception:
                with ipc.open_file(file_path) as reader:
                    table = reader.read_all()
                    
            if "spectra_tokens" in table.column_names:
                spectra_tokens = table["spectra_tokens"]
                for row in spectra_tokens:
                    tokens = row.as_py()
                    if tokens:
                        token_counts["spectra_tokens"].update(tokens)
            
            if "image_tokens" in table.column_names:
                images_tokens = table["image_tokens"]
                for row in images_tokens:
                    tokens = row.as_py()
                    if tokens:
                        token_counts["image_tokens"].update(tokens)
                        
        except Exception as e:
            print(f"Error reading {file_path}: {e}")
            
    # Plotting
    for mod_name, counts in token_counts.items():
        if not counts:
            print(f"No tokens found for {mod_name}")
            continue
            
        print(f"\n{mod_name} Stats:")
        print(f"Total tokens: {sum(counts.values())}")
        print(f"Unique tokens: {len(counts)}")
        most_common = counts.most_common(10)
        print("Top 10 most common tokens:")
        for token, count in most_common:
            print(f"  Token {token}: {count} ({count/sum(counts.values())*100:.2f}%)")
            
        tokens = list(counts.keys())
        frequencies = list(counts.values())
        
        plt.figure(figsize=(10, 6))
        if len(tokens) > 1000:
            top_counts = counts.most_common(1000)
            tokens = [t for t, c in top_counts]
            frequencies = [c for t, c in top_counts]
            
        sorted_pairs = sorted(zip(tokens, frequencies))
        tokens_sorted = [p[0] for p in sorted_pairs]
        frequencies_sorted = [p[1] for p in sorted_pairs]
        
        plt.bar(tokens_sorted, frequencies_sorted, width=1.0)
        plt.yscale('log')
        plt.title(f"Token Distribution: {mod_name}")
        plt.xlabel("Token ID")
        plt.ylabel("Frequency (log scale)")
        plt.grid(axis='y', alpha=0.3)
        plt.tight_layout()
        plot_path = os.path.join(output_dir, f"{mod_name}_distribution.png")
        plt.savefig(plot_path)
        plt.close()
        print(f"Saved distribution plot to {plot_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="/home/valonso/iac18_aasensio_shared/euclid_dr1/processed_data_arrow_interpolated_tokenized/test_0/")
    parser.add_argument("--output_dir", type=str, default="/home/valonso/iac18_mhuertas_shared/valonso/astroPT/logs/token_distributions")
    parser.add_argument("--max_files", type=int, default=377)
    args = parser.parse_args()
    
    

    
    analyze_token_distribution(args.data_dir, args.output_dir, args.max_files)
    
    