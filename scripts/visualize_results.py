import os
import glob
import re
import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict
import sys
from pathlib import Path

def process_result_file(filepath):
    """Process a single result file and extract metrics"""
    metrics = {
        'num_questions': 0,
        'prepare_time': 0,
        'semantic_similarity': 0,
        'retrieve_time': 0,
        'generate_time': 0,
        'question_similarities': [],
        'cumulative_similarities': []
    }
    
    try:
        with open(filepath, 'r') as f:
            content = f.read()
            
            # Extract all semantic similarities
            similarities = []
            
            # Pattern: [N]Average Semantic Similarity: X or [N]: Semantic Similarity: X
            sim_matches = re.findall(r'\[(\d+)\](?:Average )?Semantic Similarity: ([\d.]+)', content)
            if not sim_matches:
                # Try alternate format [N]: Semantic Similarity: X
                sim_matches = re.findall(r'\[(\d+)\]: Semantic Similarity: ([\d.]+)', content)
                
            if sim_matches:
                # Sort by question number to maintain order
                sorted_matches = sorted(sim_matches, key=lambda x: int(x[0]))
                similarities = [float(sim) for _, sim in sorted_matches]
            
            if similarities:
                metrics['question_similarities'] = similarities
                metrics['semantic_similarity'] = sum(similarities) / len(similarities)
                metrics['num_questions'] = len(similarities)
                
                # Calculate cumulative similarities
                cumulative = []
                running_sum = 0
                for i, sim in enumerate(similarities, 1):
                    running_sum += sim
                    cumulative.append(running_sum / i)
                metrics['cumulative_similarities'] = cumulative
            
            # Extract prepare time
            prepare_matches = re.findall(r'Prepare time: ([\d.]+)', content)
            if prepare_matches:
                # Take average prepare time
                metrics['prepare_time'] = sum(float(t) for t in prepare_matches) / len(prepare_matches)
            
            # Extract retrieve and generate times
            time_matches = re.findall(r'retrieve time: ([\d.]+),\s*generate time: ([\d.]+)', content)
            if time_matches:
                retrieve_times = [float(rt) for rt, _ in time_matches]
                generate_times = [float(gt) for _, gt in time_matches]
                metrics['retrieve_time'] = sum(retrieve_times) / len(retrieve_times)
                metrics['generate_time'] = sum(generate_times) / len(generate_times)
            
            print(f"\nProcessed {filepath}")
            print(f"Found {len(similarities)} similarities")
            print(f"Average similarity: {metrics['semantic_similarity']:.4f}")
            
    except Exception as e:
        print(f"Error processing {filepath}: {e}")
        return None
    
    return metrics

def generate_summaries(directory, benchmark_name):
    """Generate different types of summaries for visualization"""
    # Get all result files
    result_files = []
    
    # First try to find files in the benchmark subdirectory if it exists
    if benchmark_name:
        benchmark_pattern = f"unified_benchmark_{benchmark_name}"
        for pattern in ["result_*.txt"]:
            result_files.extend(glob.glob(os.path.join(directory, "**", benchmark_pattern, pattern), recursive=True))
    
    # If no files found, try the direct pattern
    if not result_files:
        for pattern in ["result_*.txt"]:
            result_files.extend(glob.glob(os.path.join(directory, "**", pattern), recursive=True))
    
    result_files = [f for f in result_files if "summary" not in f]
    
    if not result_files:
        print(f"No result files found in {directory}")
        return None, None
        
    # Process all files and organize results
    results = {
        'by_k': defaultdict(list),
        'by_topk': defaultdict(list),
        'combined': []  # Use a regular list for combined results
    }
    raw_results = []
    
    for filepath in result_files:
        metrics = process_result_file(filepath)
        if metrics is None:
            continue
            
        # Extract k-size from directory path and filename
        path_parts = Path(filepath).parts
        k_size = None
        
        # First try to find k-size in directory path
        for part in path_parts:
            if part.isdigit():  # Check if the directory name is a number (k-size)
                k_size = part
                break
            elif '/' in part:  # Handle Windows paths
                subparts = part.split('/')
                for subpart in subparts:
                    if subpart.isdigit():
                        k_size = subpart
                        break
                if k_size:
                    break
        
        # If not found in directory, try filename
        if k_size is None:
            filename = os.path.basename(filepath)
            # Look for patterns like "result_16_" or similar
            k_match = re.search(r'result_(\d+)_', filename)
            if k_match:
                k_size = k_match.group(1)
        
        if k_size is None:
            print(f"Could not extract k-size from {filepath}, skipping...")
            continue
        
        filename = os.path.basename(filepath)
        if "kvcache_nokv" in filename:
            method = "KV Cache (No Cache)"
            top_k = "N/A"
        elif "kvcache" in filename and "nokv" not in filename:
            method = "KV Cache"
            top_k = "N/A"
        elif "rag_Index_customgpt" in filename:
            top_k = filename.split("top")[-1].split("_")[0].split(".")[0]
            method = f"CustomGPT.ai"
        elif "rag_Index_bm25" in filename:
            top_k = filename.split("top")[-1].split("_")[0].split(".")[0]
            method = f"BM25"
        else:
            method = filename
            top_k = "N/A"
            
        metrics['k_size'] = k_size
        metrics['method'] = method
        metrics['top_k'] = top_k
        raw_results.append(metrics)
        
        # Organize results for different summary types
        results['by_k'][k_size].append(metrics)
        if top_k != "N/A":
            results['by_topk'][f"{method}_k{k_size}"].append(metrics)
        results['combined'].append(metrics)
    
    if not raw_results:
        print("No valid results found to process")
        return None, None
    
    # Generate summary files
    summaries_dir = os.path.join(directory, "summaries")
    os.makedirs(summaries_dir, exist_ok=True)
    
    # Summary by K size
    with open(os.path.join(summaries_dir, "summary_by_k.txt"), 'w') as f:
        f.write(f"Summary by Knowledge Size for {benchmark_name}\n\n")
        for k_size, metrics_list in results['by_k'].items():
            f.write(f"==> Knowledge Size: {k_size}\n")
            for m in metrics_list:
                f.write(f"Method: {m['method']} (top-k={m['top_k']})\n")
                f.write(f"Semantic Similarity: {m['semantic_similarity']}\n")
                f.write(f"Prepare Time: {m['prepare_time']}\n")
                f.write(f"Generate Time: {m['generate_time']}\n\n")
    
    # Summary by top-k
    with open(os.path.join(summaries_dir, "summary_by_topk.txt"), 'w') as f:
        f.write(f"Summary by Top-K for {benchmark_name}\n\n")
        for method_k, metrics_list in results['by_topk'].items():
            f.write(f"==> {method_k}\n")
            for m in metrics_list:
                f.write(f"Top-K: {m['top_k']}\n")
                f.write(f"Semantic Similarity: {m['semantic_similarity']}\n")
                f.write(f"Prepare Time: {m['prepare_time']}\n")
                f.write(f"Generate Time: {m['generate_time']}\n\n")
    
    # Combined summary
    with open(os.path.join(summaries_dir, "summary_combined.txt"), 'w') as f:
        f.write(f"Combined Summary for {benchmark_name}\n\n")
        for m in results['combined']:
            f.write(f"==> {m['method']} (k={m['k_size']}, top-k={m['top_k']})\n")
            f.write(f"Semantic Similarity: {m['semantic_similarity']}\n")
            f.write(f"Prepare Time: {m['prepare_time']}\n")
            f.write(f"Generate Time: {m['generate_time']}\n\n")
    
    return results, raw_results

def sanitize_filename(filename):
    """Sanitize filename by replacing invalid characters"""
    # Replace invalid characters with underscores
    invalid_chars = ['<', '>', ':', '"', '/', '\\', '|', '?', '*']
    for char in invalid_chars:
        filename = filename.replace(char, '_')
    return filename

def plot_question_similarities(metrics, output_path, method_name):
    """Create a plot showing per-question similarities with cumulative line"""
    if not metrics['question_similarities'] or not metrics['cumulative_similarities']:
        print(f"Skipping plot for {method_name} - no similarity data available")
        return
        
    plt.figure(figsize=(15, 6))
    
    # Plot individual question similarities as bars
    x = range(len(metrics['question_similarities']))
    plt.bar(x, metrics['question_similarities'], alpha=0.5, label='Per-question Similarity')
    
    # Plot cumulative similarity as a line
    if len(metrics['cumulative_similarities']) != len(metrics['question_similarities']):
        print(f"Warning: Mismatch in lengths - questions: {len(metrics['question_similarities'])}, cumulative: {len(metrics['cumulative_similarities'])}")
        # Use the shorter length to avoid dimension mismatch
        min_len = min(len(metrics['question_similarities']), len(metrics['cumulative_similarities']))
        x = range(min_len)
        plt.plot(x, metrics['cumulative_similarities'][:min_len], 'r-', label='Cumulative Similarity', linewidth=2)
    else:
        plt.plot(x, metrics['cumulative_similarities'], 'r-', label='Cumulative Similarity', linewidth=2)
    
    plt.title(f'Semantic Similarity Progression - {method_name}')
    plt.xlabel('Question Number')
    plt.ylabel('Semantic Similarity')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    
    plt.tight_layout()
    
    # Ensure the output directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path)
    plt.close()

def plot_metrics(results, output_dir, dataset_name, plot_times=False, target_topk=None):
    """Create visualization plots for different metrics and comparisons"""
    os.makedirs(output_dir, exist_ok=True)
    
    # Sort k sizes and convert to integers for proper sorting
    k_sizes = sorted([int(k) for k in results['by_k'].keys()])
    
    # Define method order and colors
    method_order = ['BM25', 'KV Cache', 'KV Cache (No Cache)', 'CustomGPT.ai', 'OpenAI']
    method_colors = {
        'BM25': '#1f77b4',      # Blue
        'KV Cache': '#2ca02c',   # Green
        'KV Cache (No Cache)': '#d62728',  # Red
        'CustomGPT.ai': '#ff7f0e',  # Orange
        'OpenAI': '#9467bd'      # Purple
    }
    
    # Prepare data for plotting
    plot_data = {k: {method: 0 for method in method_order} for k in k_sizes}
    
    # Fill in the data
    for k in k_sizes:
        metrics_list = results['by_k'][str(k)]
        method_values = defaultdict(list)
        
        for metrics in metrics_list:
            # Skip if not matching target_topk (if specified)
            if target_topk is not None and metrics['top_k'] != str(target_topk):
                continue
                
            method = metrics['method']
            if method not in method_order:
                print(f"Warning: Unknown method {method}, skipping...")
                continue
            method_values[method].append(metrics['semantic_similarity'])
        
        for method in method_values:
            if method_values[method]:  # Only calculate average if we have values
                plot_data[k][method] = sum(method_values[method]) / len(method_values[method])
    
    # Create the plot
    plt.figure(figsize=(12, 6))
    
    # Calculate bar positions
    x = np.arange(len(k_sizes))
    width = 0.25
    multiplier = 0
    
    # Plot bars in specified order
    for method in method_order:
        values = [plot_data[k][method] for k in k_sizes]
        # Only plot if we have non-zero values
        if any(values):
            offset = width * multiplier
            rects = plt.bar(x + offset, values, width, label=method, color=method_colors[method])
            plt.bar_label(rects, fmt='%.3f')
            multiplier += 1
    
    # Customize the plot
    plt.xlabel('Knowledge Size')
    plt.ylabel('Average Semantic Similarity')
    title = f'Average Semantic Similarity by Knowledge Size - {dataset_name}'
    if target_topk is not None:
        title += f' (top-k={target_topk})'
    plt.title(title)
    
    # Set x-axis labels with k= prefix
    plt.xticks(x + width, [f'k={k}' for k in k_sizes])
    
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.ylim(0, 1.0)
    
    # Adjust layout to prevent label cutoff
    plt.tight_layout()
    
    # Save the plot
    output_filename = 'similarity_by_k.png'
    if target_topk is not None:
        output_filename = f'similarity_by_k_top{target_topk}.png'
    plt.savefig(os.path.join(output_dir, output_filename), bbox_inches='tight', dpi=300)
    plt.close()
    
    # Plot comparisons by top-k for each method and k size
    for method_k, metrics_list in results['by_topk'].items():
        plt.figure(figsize=(10, 6))
        top_k_values = sorted(set(m['top_k'] for m in metrics_list))
        values = [next((m['semantic_similarity'] for m in metrics_list if m['top_k'] == tk), 0) 
                 for tk in top_k_values]
        
        # Extract base method from method_k
        base_method = None
        if 'bm25' in method_k.lower():
            base_method = 'BM25'
        elif 'kvcache' in method_k.lower():
            base_method = 'KV Cache'
        elif 'customgpt' in method_k.lower():
            base_method = 'CustomGPT.ai'
        elif 'openai' in method_k.lower():
            base_method = 'OpenAI'
            
        if base_method and base_method in method_colors:
            plt.bar(range(len(top_k_values)), values, color=method_colors[base_method])
        else:
            plt.bar(range(len(top_k_values)), values)
            
        plt.title(f'Average Semantic Similarity by Top-K - {method_k}')
        plt.xticks(range(len(top_k_values)), [f'top-k={tk}' for tk in top_k_values], rotation=45)
        plt.ylabel('Average Semantic Similarity')
        plt.tight_layout()
        safe_method_k = sanitize_filename(method_k)
        plt.savefig(os.path.join(output_dir, f'similarity_by_topk_{safe_method_k}.png'))
        plt.close()
    
    # Plot combined method comparison
    plt.figure(figsize=(12, 6))
    
    # Prepare data for combined plot
    combined_data = defaultdict(list)
    for metrics in results['combined']:
        method = metrics['method']
        if method not in method_order:
            print(f"Warning: Unknown method {method} in combined plot, skipping...")
            continue
        top_k = metrics.get('top_k', 'N/A')
        key = f"{method} (top-k={top_k})"
        combined_data[key].append(metrics['semantic_similarity'])
    
    # Calculate averages
    averages = {k: sum(v)/len(v) for k, v in combined_data.items()}
    
    # Custom sorting function
    def sort_key(method_name):
        base_method = method_name.split(' (')[0]
        top_k = method_name.split('top-k=')[1].rstrip(')')
        
        # Primary sort by method order
        try:
            primary_key = method_order.index(base_method)
        except ValueError:
            primary_key = len(method_order)  # Put unknown methods at the end
        
        # Secondary sort by top-k
        if top_k == 'N/A':
            secondary_key = 999  # Put N/A at the end
        else:
            secondary_key = int(top_k)
            
        return (primary_key, secondary_key)
    
    # Sort methods using custom sorting
    sorted_methods = sorted(averages.keys(), key=sort_key)
    values = [averages[method] for method in sorted_methods]
    
    # Create bars with colors based on method
    bars = plt.bar(range(len(sorted_methods)), values)
    
    # Color each bar based on its method
    for bar, method_name in zip(bars, sorted_methods):
        base_method = method_name.split(' (')[0]
        if base_method in method_colors:
            bar.set_color(method_colors[base_method])
    
    # Customize the plot
    plt.xticks(range(len(sorted_methods)), sorted_methods, rotation=45, ha='right')
    plt.ylabel('Average Semantic Similarity')
    plt.title(f'Combined Method Comparison - Average Semantic Similarity - {dataset_name}')
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.ylim(0, 1.0)
    
    # Add value labels on top of bars
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.3f}',
                ha='center', va='bottom')
    
    # Adjust layout to prevent label cutoff
    plt.tight_layout()
    
    # Save the plot
    plt.savefig(os.path.join(output_dir, 'similarity_combined.png'), bbox_inches='tight', dpi=300)
    plt.close()
    
    # Plot time-based metrics if requested
    if plot_times:
        for metric in ['prepare_time', 'generate_time', 'retrieve_time']:
            plt.figure(figsize=(12, 8))
            for i, method in enumerate(method_order):
                values = []
                for k in k_sizes:
                    method_metrics = [m for m in results['by_k'][str(k)] if m['method'] == method]
                    if method_metrics:
                        values.append(method_metrics[0].get(metric, 0))
                    else:
                        values.append(0)
                
                if any(values):  # Only plot if we have non-zero values
                    x = np.arange(len(k_sizes)) + i * 0.25
                    plt.bar(x, values, 0.25, label=method, color=method_colors[method])
            
            plt.title(f'{metric.replace("_", " ").title()} by Knowledge Size - {dataset_name}')
            plt.xticks(range(len(k_sizes)), [f'k={k}' for k in k_sizes])
            plt.ylabel('Time (seconds)')
            plt.legend(bbox_to_anchor=(1.05, 1))
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, f'{metric}_by_k.png'), bbox_inches='tight')
            plt.close()

def generate_aggregate_summary(results, summaries_dir, dataset_name):
    """Generate aggregate summaries across all k sizes"""
    os.makedirs(summaries_dir, exist_ok=True)
    
    # Summary by K size
    with open(os.path.join(summaries_dir, "aggregate_summary_by_k.txt"), 'w') as f:
        f.write(f"Aggregate Summary by Knowledge Size for {dataset_name}\n\n")
        for k_size, metrics_list in sorted(results['by_k'].items(), key=lambda x: int(x[0])):
            f.write(f"==> Knowledge Size: {k_size}\n")
            # Group by method and top-k
            method_groups = defaultdict(list)
            for m in metrics_list:
                key = f"{m['method']} (top-k={m['top_k']})"
                method_groups[key].append({
                    'semantic_similarity': m['semantic_similarity'],
                    'prepare_time': m['prepare_time'],
                    'generate_time': m['generate_time']
                })
            
            # Calculate averages for each method
            for method_key, metrics in sorted(method_groups.items()):
                avg_similarity = sum(m['semantic_similarity'] for m in metrics) / len(metrics)
                avg_prepare = sum(m['prepare_time'] for m in metrics) / len(metrics)
                avg_generate = sum(m['generate_time'] for m in metrics) / len(metrics)
                
                f.write(f"Method: {method_key}\n")
                f.write(f"Average Semantic Similarity: {avg_similarity:.4f}\n")
                f.write(f"Average Prepare Time: {avg_prepare:.4f}\n")
                f.write(f"Average Generate Time: {avg_generate:.4f}\n")
                f.write(f"Number of runs: {len(metrics)}\n\n")

    # Summary by method
    with open(os.path.join(summaries_dir, "aggregate_summary_by_method.txt"), 'w') as f:
        f.write(f"Aggregate Summary by Method for {dataset_name}\n\n")
        method_groups = defaultdict(list)
        for m in results['combined']:
            key = f"{m['method']} (top-k={m['top_k']})"
            method_groups[key].append(m)
        
        for method_key, metrics_list in sorted(method_groups.items()):
            f.write(f"==> {method_key}\n")
            avg_similarity = sum(m['semantic_similarity'] for m in metrics_list) / len(metrics_list)
            avg_prepare = sum(m['prepare_time'] for m in metrics_list) / len(metrics_list)
            avg_generate = sum(m['generate_time'] for m in metrics_list) / len(metrics_list)
            
            f.write(f"Average Semantic Similarity: {avg_similarity:.4f}\n")
            f.write(f"Average Prepare Time: {avg_prepare:.4f}\n")
            f.write(f"Average Generate Time: {avg_generate:.4f}\n")
            f.write(f"Number of runs: {len(metrics_list)}\n")
            f.write(f"K sizes tested: {sorted(set(m['k_size'] for m in metrics_list))}\n\n")

def main():
    if len(sys.argv) > 1:
        base_dir = sys.argv[1]
        # Make benchmark_name optional
        benchmark_name = sys.argv[2] if len(sys.argv) > 2 else None
        target_topk = sys.argv[3] if len(sys.argv) > 3 else None
        print(f"Processing directory: {base_dir}")
        if benchmark_name:
            print(f"Using benchmark name: {benchmark_name}")
        if target_topk:
            print(f"Filtering for top-k={target_topk}")
        
        base_dir = os.path.abspath(base_dir)
        if not os.path.exists(base_dir):
            print(f"Error: Directory {base_dir} does not exist")
            return
        
        # Extract dataset name from path
        dataset_name = None
        path_parts = Path(base_dir).parts
        for part in path_parts:
            if part in ["hotpotqa-train", "squad"]:  # Add other dataset names as needed
                dataset_name = part
                break
        
        if not dataset_name:
            # Fallback: try to get dataset name from the immediate parent directory
            dataset_name = os.path.basename(os.path.dirname(base_dir))
            print(f"Using directory name as dataset: {dataset_name}")
        
        # Process the entire benchmark directory at once
        print(f"\nProcessing benchmark directory: {base_dir}")
        results, raw_results = generate_summaries(base_dir, benchmark_name)
        
        if results and raw_results:
            # Create output directories
            output_suffix = f"_{benchmark_name}" if benchmark_name else ""
            dataset_aggregate_dir = os.path.join(base_dir, f"aggregate_results{output_suffix}")
            dataset_vis_dir = os.path.join(dataset_aggregate_dir, "visualizations")
            dataset_summaries_dir = os.path.join(dataset_aggregate_dir, "summaries")
            
            os.makedirs(dataset_vis_dir, exist_ok=True)
            os.makedirs(dataset_summaries_dir, exist_ok=True)
            
            # Generate overall visualizations
            plot_metrics(results, dataset_vis_dir, dataset_name, target_topk=target_topk)
            
            # Generate overall summaries
            generate_aggregate_summary(results, dataset_summaries_dir, dataset_name)
            
            print(f"Overall aggregate results saved to: {dataset_aggregate_dir}")
            
            # Generate per-k visualizations and summaries
            for k_size in results['by_k'].keys():
                k_vis_dir = os.path.join(dataset_vis_dir, f"k{k_size}")
                k_summaries_dir = os.path.join(dataset_summaries_dir, f"k{k_size}")
                
                os.makedirs(k_vis_dir, exist_ok=True)
                os.makedirs(k_summaries_dir, exist_ok=True)
                
                # Create k-specific results structure
                k_results = {
                    'by_k': {k_size: results['by_k'][k_size]},
                    'by_topk': {},
                    'combined': [m for m in results['combined'] if m['k_size'] == k_size]
                }
                
                # Filter topk results for this k-size
                for method_k, metrics_list in results['by_topk'].items():
                    if f"k{k_size}" in method_k:
                        k_results['by_topk'][method_k] = metrics_list
                
                # Generate visualizations for this k-size
                plot_metrics(k_results, k_vis_dir, f"{dataset_name} (k={k_size})", target_topk=target_topk)
                
                # Generate per-method question similarity plots
                for metrics in [m for m in raw_results if m['k_size'] == k_size]:
                    method_name = f"{metrics['method']}_k{metrics['k_size']}_top{metrics['top_k']}"
                    safe_method_name = sanitize_filename(method_name)
                    output_path = os.path.join(k_vis_dir, f'question_similarities_{safe_method_name}.png')
                    plot_question_similarities(metrics, output_path, method_name)
                
                # Generate summaries for this k-size
                generate_aggregate_summary(k_results, k_summaries_dir, f"{dataset_name} (k={k_size})")
        else:
            print("No results found to process!")
    else:
        print("Please provide a base directory as an argument")

if __name__ == "__main__":
    main() 