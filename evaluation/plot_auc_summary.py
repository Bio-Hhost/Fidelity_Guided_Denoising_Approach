"""
Generates a summary plot of AUC vs. Noise Scale for all methods.

This script recursively scans the output directory from 'evaluate_detection_threshold_scan.py', 
finds all 'detection_metrics_all_variants.csv' files, 
and extracts the Area Under the Curve (AUC) for each denoising method at each noise level.

It then generates a single line plot summarizing the performance (AUC) of all methods as a function of the noise scale.
"""

import pandas as pd
import glob
import os
import sys
import matplotlib.pyplot as plt
import seaborn as sns
import re
import argparse

print("Starting AUC summary generation for all denoising variants...")

import sys as _sys
_sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "figures"))
from figure_style import style_for_data_type as get_method_style

def main(args):
    search_pattern = os.path.join(args.input_dir, '**', 'detection_metrics_all_variants.csv')
    result_files = glob.glob(search_pattern, recursive=True)

    if not result_files:
        print(f"Error: No 'detection_metrics_all_variants.csv' files found in {args.input_dir}.")
        sys.exit(1)

    print(f"Found {len(result_files)} result files to summarize.")

    summary_data = []
    for f_path in result_files:
        try:
            parts = f_path.split(os.sep)
            scale_info = parts[-2] 

            scale_match = re.search(r'(\d+\.\d+)', scale_info)
            if not scale_match:
                print(f"Warning: Could not parse scale from directory {scale_info}. Skipping file {f_path}")
                continue
            
            scale_numeric = float(scale_match.group(1))
            
            df = pd.read_csv(f_path)
            
            auc_data = df.groupby('data_type')['AUC'].first()

            for data_type, auc_value in auc_data.items():
                summary_data.append({
                    'scale_numeric': scale_numeric,
                    'data_type': data_type, 
                    'auc': auc_value
                })

        except (IndexError, pd.errors.EmptyDataError, KeyError, TypeError) as e:
            print(f"Warning: Could not process file {f_path}. It might be empty or malformed. Error: {e}")
            continue

    if not summary_data:
        print("Error: No valid data could be extracted from the result files.")
        sys.exit(1)

    summary_df = pd.DataFrame(summary_data)

    style_info = summary_df['data_type'].apply(get_method_style)
    summary_df['Method'] = style_info.apply(lambda x: x[0])
    summary_df['Color'] = style_info.apply(lambda x: x[1])

    summary_df = summary_df.sort_values('scale_numeric').reset_index(drop=True)

    print("Generating summary plot...")
    plt.style.use('seaborn-v0_8-whitegrid')
    fig, ax = plt.subplots(figsize=(14, 9))

    palette = {row['Method']: row['Color'] for index, row in summary_df.drop_duplicates('Method').iterrows()}

    sns.lineplot(
        data=summary_df,
        x='scale_numeric',
        y='auc',
        hue='Method',
        palette=palette,
        marker='o',
        ax=ax
    )

    ax.set_title('Denoising Method Performance (AUC) vs. Noise Scale', fontsize=18, weight='bold')
    ax.set_ylabel('Area Under PR Curve (AUC)', fontsize=14)
    ax.set_xlabel('Simulated Noise Scale (Higher is more noise)', fontsize=14)
    ax.set_ylim(-0.05, 1.05)
    ax.grid(True, which='both', linestyle='--', linewidth=0.5)

    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles=handles, labels=labels, title='Method', bbox_to_anchor=(1.02, 1), loc='upper left')

    plt.tight_layout(rect=[0, 0, 0.85, 1]) 

    try:
        os.makedirs(os.path.dirname(args.output_plot), exist_ok=True)
        plt.savefig(args.output_plot, dpi=300)
        print(f"\nSummary plot successfully saved to: {args.output_plot}")
    except Exception as e:
        print(f"\nError saving plot: {e}")

def parse_args():
    parser = argparse.ArgumentParser(description="Generate a summary plot of AUC vs. Noise Scale from evaluation results.")
    
    parser.add_argument("--input_dir", type=str, required=True,
                        help="Path to the base directory where 'evaluate_detections.py' saved its results (e.g., 'Cy3_Best/Evaluation_Results').")
    parser.add_argument("--output_plot", type=str, required=True,
                        help="Path to save the final summary .png plot (e.g., 'Cy3_Best/Evaluation_Results/AUC_Summary_Plot.png').")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    main(args)
