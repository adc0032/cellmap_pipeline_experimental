#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug 21 18:10:51 2025

@author: Amanda D. Clark (clarkad@uab.edu)
"""

"""
CM4AI Parameter Testing Visualization

Program for analyzing multi-parameter optimization results 
from cellmap pipeline.

Usage:
    python plot_eval_hierarchy.py --input results.csv --output ./plots/
    
    where results.csv is output of results from a paramater search space run
    for generated hierarchies
    
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import argparse
import os
import re
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any, Union
from scipy import stats
from scipy.stats import kruskal, mannwhitneyu
from statsmodels.stats.multitest import multipletests
import warnings
warnings.filterwarnings('ignore')

# expected input col names from csv
METRICS = [
    'hierarchy_mean_corum_jaccard',
    'hierarchy_mean_go_cc_jaccard',
    'hierarchy_mean_hpa_jaccard'
    ]

METRIC_LABELS = {
    'hierarchy_mean_corum_jaccard': 'CORUM complexes',
    'hierarchy_mean_go_cc_jaccard': 'GO Cellular Components',
    'hierarchy_mean_hpa_jaccard': 'HPA Subcellular Localization'
    }

PARAM_COLS = [
    'coembed_algorithm',
    'hierarchy_k',
    'hierarchy_maxres',
    'hierarchy_containment_threshold',
    'hierarchy_jaccard_threshold',
    'hierarchy_min_diff',
    'hierarchy_min_system_size',
    'hierarchy_ppi_cutoffs',
    'hierarchy_parent_ppi_cutoffs',
    'hierarchy_bootstrap_edges'
    ]

ALGORITHM_COL = 'hierarchy_algorithm'

# plot params
plt.style.use('seaborn-v0_8-darkgrid')
FIGSIZE = (12,8)
DPI = 300

COLORS = {
    'leiden': '#3498db',
    'lovain': '#e74c3c',
    'walktrap': '#2ecc71'
    }

def load_data(filepath: str) -> pd.DataFrame:
    """Load csv data for hierarchy performance comparison"""
    data = pd.read_csv(filepath)
    print(f"Loaded hierarchy evaluation results from {len(data)} trials.")
    
    available_metrics = [col for col in METRICS if col in data.columns]
    if not available_metrics:
        raise ValueError(f"No metric columns found. Expected: {METRICS}")
        
    varying_params = []
    for param in PARAM_COLS:
        if param in data.columns and data[param].nunique() > 1:
            varying_params.append(param)
            
    print(f"Found metrics: {available_metrics}")
    print(f"Tested parameters: {varying_params}")
    
    return data, varying_params, available_metrics

def stats_analysis(
        data: pd.DataFrame, 
        varying_params: List[str],
        available_metrics: List[str]) -> dict:
    """Statistical analysis for jaccard metrics"""
    
    results = {}
    
    for metric in available_metrics:
        metric_results = {}
        
        if ALGORITHM_COL in data.columns:
            algorithms = data[ALGORITHM_COL].unique()
            if len(algorithms) > 2:
                # run Kruskal-Wallis
                groups = [data[data[ALGORITHM_COL] == alg][metric].dropna()
                          for alg in algorithms]
                stat, p = kruskal(*groups)
                metric_results['algorithm_test'] = {
                    'test': 'Kruskal-Wallis',
                    'statistic': stat,
                    'p_value': p
                    }
            elif len(algorithms) == 2:
                # run Mann-Whitney U
                g1, g2 = [data[data[ALGORITHM_COL] == alg][metric].dropna()
                          for alg in algorithms]
                stat, p = mannwhitneyu(g1, g2, alternative='two-sided')
                metric_results['algorithm_test'] = {
                    'test': 'Mann-Whitney U',
                    'statistic': stat,
                    'p_value': p
                    }
        # pearson correlation for parameters
        numeric_params = [p for p in varying_params 
                          if pd.api.types.is_numeric_dtype(data[p])]
        if numeric_params:
            corr_results = []
            for param in numeric_params:
                r, p = stats.pearsonr(data[param], data[metric])
                corr_results.append(
                    {
                        'parameter': param, 
                        'correlation': r,
                        'p_value': p
                        }
                    )
            p_vals = [r['p_value'] for r in corr_results]
            p_adjust = multipletests(p_vals, method='bonferroni')[1]
            for i, result in enumerate(corr_results):
                result['p_adjusted'] = p_adjust[i]
            metric_results['correlations'] = corr_results
        results[metric] = metric_results
    return results

def create_parameter_effects(data: pd.DataFrame, varying_params: List[str], available_metrics: List[str], save_path: Optional[str] = None):
    """Analyze parameter effects on each metric separately."""
    if not varying_params:
        print("No varying parameters found")
        return None
    
    n_params = len(varying_params)
    n_metrics = len(available_metrics)
    
    y_max = np.ceil(data[available_metrics].max().max() * 10) / 10

    fig, axes = plt.subplots(n_metrics, n_params, figsize=(6 * n_params, 5 * n_metrics))
    if n_metrics == 1 and n_params == 1:
        axes = [[axes]]
    elif n_metrics == 1:
        axes = [axes]
    elif n_params == 1:
        axes = [[ax] for ax in axes]
    
    algorithms = data[ALGORITHM_COL].unique() if ALGORITHM_COL in data.columns else [None]

    for metric_idx, metric in enumerate(available_metrics):
        for param_idx, param in enumerate(varying_params):
            ax = axes[metric_idx][param_idx]
            
            if len(algorithms) > 1 and algorithms[0] is not None:
                # Plot by algorithm
                for alg in algorithms:
                    subset = data[data[ALGORITHM_COL] == alg]
                    if len(subset) > 0:
                        param_stats = subset.groupby(param)[metric].agg(['mean', 'std'])
                        color = COLORS.get(alg.lower(), 'blue')
                        ax.errorbar(param_stats.index, param_stats['mean'], 
                                   yerr=param_stats['std'], 
                                   marker='o', label=alg, color=color,
                                   linewidth=2, markersize=6)
                if metric_idx == 0:  # Only show legend on top row
                    ax.legend()
            else:
                # Single line
                param_stats = data.groupby(param)[metric].agg(['mean', 'std'])
                ax.errorbar(param_stats.index, param_stats['mean'], 
                           yerr=param_stats['std'], 
                           marker='o', color='blue', linewidth=2, markersize=6)
            
            if metric_idx == 2:
                ax.set_xlabel(param.replace('_', ' ').title(), fontweight='bold')
            ax.set_ylim(0, y_max)
            # Y-axis labels
            if param_idx == 0:  # Only leftmost column
                metric_label = METRIC_LABELS.get(metric, metric.replace("_", " ").title())
                ax.set_ylabel(f'{metric_label}\nJaccard Index', fontweight='bold')
            
            # Title only on top row
            if metric_idx == 0:
                ax.set_title(f'Effect of {param.replace("_", " ").title()}', fontweight='bold')
            
            ax.grid(True, alpha=0.3)
            
            # Rotate labels if text
            if data[param].dtype == 'object':
                ax.tick_params(axis='x', rotation=45)
    
    plt.suptitle('Parameter Effects on Each Metric', fontsize=16, fontweight='bold')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=DPI, bbox_inches='tight')
    
    return fig

def create_performance_summary(data: pd.DataFrame, varying_params: List[str], available_metrics: List[str], stats_results: dict, save_path: Optional[str] = None):
    """Create performance summary for each metric separately."""
    
    y_max = np.ceil(data[available_metrics].max().max() * 10) / 10

    n_metrics = len(available_metrics)
    fig, axes = plt.subplots(2, n_metrics, figsize=(6 * n_metrics, 13))
    if n_metrics == 1:
        axes = axes.reshape(-1, 1)
    
    for i, metric in enumerate(available_metrics):
        metric_label = METRIC_LABELS.get(metric, metric.replace("_", " ").title())
        
        # Top row: Metric correlations
        ax = axes[0, i]
    
        # Create scatter plot of this metric vs other metrics
        other_metrics = [m for m in available_metrics if m != metric]
        
        if len(other_metrics) >= 1:
            other_metric = other_metrics[0]
            ax.scatter(data[metric], data[other_metric], alpha=0.6)
            
            # Add correlation coefficient
            r, p = stats.pearsonr(data[metric], data[other_metric])
            ax.text(0.05, 0.95, f'r = {r:.3f}\np = {p:.3f}', 
                    transform=ax.transAxes, bbox=dict(boxstyle='round', facecolor='white'))
            
            metric_label1 = METRIC_LABELS.get(metric, metric)
            metric_label2 = METRIC_LABELS.get(other_metric, other_metric)
            ax.set_xlabel(f'{metric_label1} Jaccard')
            ax.set_ylabel(f'{metric_label2} Jaccard')
            ax.set_title(f'{metric_label1} vs {metric_label2}', fontweight='bold')
            
        # Bottom row: Algorithm comparison
        ax2 = axes[1, i]
        if ALGORITHM_COL in data.columns:
            algo_stats = data.groupby(ALGORITHM_COL)[metric].agg(['mean', 'std', 'count'])
            bars = ax2.bar(range(len(algo_stats)), algo_stats['mean'],
                          yerr=algo_stats['std'], capsize=5, alpha=0.7)
            
            # Color bars
            for bar, alg in zip(bars, algo_stats.index):
                bar.set_color(COLORS.get(alg.lower(), 'gray'))
            
            ax2.set_xlabel('Algorithm', fontweight='bold')
            ax2.set_ylabel('Jaccard Index', fontweight='bold')
            ax2.set_title(f'{metric_label} by Algorithm', fontweight='bold')
            ax2.set_ylim(0, y_max)
            ax2.set_xticks(range(len(algo_stats)))
            ax2.set_xticklabels(algo_stats.index, rotation=45)
            ax2.grid(True, alpha=0.3)
            
            # Add significance stars
            if metric in stats_results and 'algorithm_test' in stats_results[metric]:
                p_val = stats_results[metric]['algorithm_test']['p_value']
                sig = '***' if p_val < 0.001 else '**' if p_val < 0.01 else '*' if p_val < 0.05 else 'ns'
                test_name = stats_results[metric]['algorithm_test']['test']
                ax2.text(0.02, 0.98, f"{test_name}\np={p_val:.4f} {sig}", 
                        transform=ax2.transAxes, va='top', ha='left', 
                        bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    plt.suptitle('Performance Summary', fontsize=16, fontweight='bold')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=DPI, bbox_inches='tight')
    
    return fig

def get_best_configs(data: pd.DataFrame, metric: str, n: int = 5) -> pd.DataFrame:
    """Get top N configurations for input metrics."""
    return data.nlargest(n, metric)[
        [ALGORITHM_COL] + PARAM_COLS + [metric]
        
    ]

def generate_report(data: pd.DataFrame, varying_params: List[str], available_metrics: List[str], stats_results: dict, output_dir: str):
    """Generate comprehensive report with separate metric analysis."""
    report_path = os.path.join(output_dir, 'analysis_report.txt')
    
    with open(report_path, 'w') as f:
        f.write("CM4AI Parameter Analysis Report - Separate Metrics\n")
        f.write("=" * 60 + "\n\n")
        
        # Data summary
        f.write(f"Dataset: {len(data)} configurations analyzed\n")
        f.write(f"Metrics analyzed separately: {len(available_metrics)}\n")
        for metric in available_metrics:
            metric_label = METRIC_LABELS.get(metric, metric)
            f.write(f"  - {metric_label}\n")
        f.write(f"Varying parameters: {varying_params}\n\n")
        
        # Performance summary for each metric
        for metric in available_metrics:
            metric_label = METRIC_LABELS.get(metric, metric)
            f.write(f"{metric_label} Performance:\n")
            f.write(f"  Mean: {data[metric].mean():.4f}\n")
            f.write(f"  Std:  {data[metric].std():.4f}\n")
            f.write(f"  Min:  {data[metric].min():.4f}\n")
            f.write(f"  Max:  {data[metric].max():.4f}\n\n")
        
        # Statistical results for each metric
        f.write("Statistical Analysis by Metric:\n")
        f.write("-" * 40 + "\n")
        for metric in available_metrics:
            metric_label = METRIC_LABELS.get(metric, metric)
            f.write(f"\n{metric_label}:\n")
            
            if metric in stats_results and 'algorithm_test' in stats_results[metric]:
                test = stats_results[metric]['algorithm_test']
                f.write(f"  Algorithm comparison ({test['test']}): p={test['p_value']:.4f}\n")
            
            if metric in stats_results and 'correlations' in stats_results[metric]:
                f.write(f"  Parameter Correlations (Bonferroni corrected):\n")
                for corr in stats_results[metric]['correlations']:
                    sig = '*' if corr['p_adjusted'] < 0.05 else ''
                    f.write(f"    {corr['parameter']}: r={corr['correlation']:.3f}, p={corr['p_adjusted']:.4f}{sig}\n")
        
        # Best configurations for each metric
        f.write(f"\nTop 3 Configurations by Metric:\n")
        f.write("-" * 40 + "\n")
        for metric in available_metrics:
            metric_label = METRIC_LABELS.get(metric, metric)
            f.write(f"\n{metric_label}:\n")
            best_configs = get_best_configs(data, metric, 3)
            for i, (idx, row) in enumerate(best_configs.iterrows()):
                f.write(f"  {i+1}. {metric_label}: {row[metric]:.4f}\n")
                for param in PARAM_COLS:
                    if param in row:
                        f.write(f"     {param}: {row[param]}\n")
        
        # Algorithm comparison for each metric
        if ALGORITHM_COL in data.columns:
            f.write(f"\nAlgorithm Performance by Metric:\n")
            f.write("-" * 40 + "\n")
            for metric in available_metrics:
                metric_label = METRIC_LABELS.get(metric, metric)
                f.write(f"\n{metric_label}:\n")
                algo_stats = data.groupby(ALGORITHM_COL)[metric].agg(['mean', 'std', 'count'])
                for alg, stats in algo_stats.iterrows():
                    f.write(f"  {alg}: {stats['mean']:.4f} ± {stats['std']:.4f} (n={stats['count']})\n")
    
    print(f"Report saved to: {report_path}")
    return stats_results

def main():
    """Main execution function."""
    parser = argparse.ArgumentParser(description='CM4AI Hierarchy Parameter Analysis')
    parser.add_argument('--input', '-i', required=True, help='Input CSV file')
    parser.add_argument('--output', '-o', default='./plots/', help='Output directory')
    
    args = parser.parse_args()
    
    print(f"Loading data from: {args.input}")
    data, varying_params, available_metrics = load_data(args.input)
    os.makedirs(args.output, exist_ok=True)
    
    print("Generating visualizations...")
    create_parameter_effects(data, varying_params, available_metrics, save_path=os.path.join(args.output, 'parameter_effects.png'))
    
    print("Running statistical analysis...")
    stats_results = stats_analysis(data, varying_params, available_metrics)
    perf_plot_path = os.path.join(args.output, 'performance_summary.png')
    create_performance_summary(data, varying_params, available_metrics, stats_results, save_path=perf_plot_path)
    generate_report(data, varying_params, available_metrics, stats_results, args.output)
    
    # Print statistical summaries
    print("\nStatistical Summary by Metric:")
    print("-" * 40)
    for metric in available_metrics:
        metric_label = METRIC_LABELS.get(metric, metric)
        print(f"\n{metric_label}:")
        
        if metric in stats_results and 'algorithm_test' in stats_results[metric]:
            test = stats_results[metric]['algorithm_test']
            print(f"  Algorithm test: {test['test']} p={test['p_value']:.4f}")
        
        if metric in stats_results and 'correlations' in stats_results[metric]:
            sig_corrs = [c for c in stats_results[metric]['correlations'] if c['p_adjusted'] < 0.05]
            if sig_corrs:
                print(f"  Significant correlations:")
                for corr in sig_corrs:
                    print(f"    {corr['parameter']}: r={corr['correlation']:.3f}")

    
    # Print best configurations
    print(f"\nBest configuration for each metric:")
    print("-" * 40)
    for metric in available_metrics:
        metric_label = METRIC_LABELS.get(metric, metric)
        best = get_best_configs(data, metric, 1)
        print(f"\n{metric_label}: {best[metric].iloc[0]:.4f}")
        print(f"  Algorithm: {best[ALGORITHM_COL].iloc[0] if ALGORITHM_COL in best.columns else 'N/A'}")
    
    print(f"\n Analysis complete! Results saved to: {args.output}")

if __name__ == "__main__":
    main()
