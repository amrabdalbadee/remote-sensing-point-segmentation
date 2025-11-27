"""
Enhanced Automated Experiment Runner with Publication-Ready Results

This script runs comprehensive experiments and generates well-organized results
suitable for technical report writing.

Usage:
    python scripts/run_experimentspy --experiment exp1
    python scripts/run_experiments.py --experiment exp2
    python scripts/run_experiments.py --experiment all
"""
import os
import sys
import argparse
import yaml
import subprocess
from pathlib import Path
import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

# Set publication-quality plot defaults
plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['font.size'] = 12
plt.rcParams['axes.labelsize'] = 14
plt.rcParams['axes.titlesize'] = 16
plt.rcParams['legend.fontsize'] = 12
plt.rcParams['figure.figsize'] = (10, 6)
sns.set_style("whitegrid")
sns.set_palette("husl")


class EnhancedExperimentRunner:
    """Enhanced experiment runner with publication-ready outputs"""
    
    def __init__(self, base_config_path, output_base_dir):
        self.base_config = self._load_config(base_config_path)
        self.output_base_dir = Path(output_base_dir)
        self.output_base_dir.mkdir(parents=True, exist_ok=True)
        
        # Create organized structure
        self.results_dir = self.output_base_dir / 'results'
        self.figures_dir = self.output_base_dir / 'figures'
        self.tables_dir = self.output_base_dir / 'tables'
        self.checkpoints_dir = self.output_base_dir / 'checkpoints'
        
        for dir_path in [self.results_dir, self.figures_dir, self.tables_dir, self.checkpoints_dir]:
            dir_path.mkdir(parents=True, exist_ok=True)
        
        self.results = []
        
        # Create experiment log
        self.log_file = self.output_base_dir / 'experiment_log.txt'
        self._log(f"Experiment started at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    def _log(self, message):
        """Log messages to file and console"""
        print(message)
        with open(self.log_file, 'a') as f:
            f.write(f"[{datetime.now().strftime('%H:%M:%S')}] {message}\n")
    
    def _load_config(self, config_path):
        with open(config_path, 'r') as f:
            return yaml.safe_load(f)
    
    def _save_config(self, config, path):
        with open(path, 'w') as f:
            yaml.dump(config, f, default_flow_style=False, sort_keys=False)
    
    def run_experiment(self, exp_name, config_overrides):
        """Run a single experiment"""
        self._log(f"\n{'='*70}")
        self._log(f"Running: {exp_name}")
        self._log(f"{'='*70}")
        
        # Create experiment config
        exp_config = yaml.safe_load(yaml.dump(self.base_config))
        
        # Apply overrides
        for key_path, value in config_overrides.items():
            keys = key_path.split('.')
            config_section = exp_config
            for key in keys[:-1]:
                config_section = config_section[key]
            config_section[keys[-1]] = value
        
        # Update experiment name and directory
        exp_config['experiment']['name'] = exp_name
        exp_dir = self.checkpoints_dir / exp_name
        exp_config['experiment']['output_dir'] = str(exp_dir)
        
        # Save experiment config
        exp_dir.mkdir(parents=True, exist_ok=True)
        config_path = exp_dir / 'config.yaml'
        self._save_config(exp_config, config_path)
        
        self._log(f"Configuration saved to: {config_path}")
        self._log(f"Overrides: {config_overrides}")
        
        # Run training
        train_script = Path(__file__).parent / 'train.py'
        cmd = ['python', str(train_script), '--config', str(config_path)]
        
        try:
            self._log("Starting training...")
            # Let stdout/stderr stream directly to the terminal so progress is visible
            result = subprocess.run(cmd, check=True)
            self._log("Training completed successfully!")
            
            # Extract results
            self._extract_results(exp_name, exp_dir, config_overrides)
            return True
            
        except subprocess.CalledProcessError as e:
            self._log(f"ERROR: Training failed!")
            self._log(f"Error output: {e.stderr}")
            return False
    
    def _extract_results(self, exp_name, exp_dir, config_overrides):
        """Extract and store results from completed experiment"""
        history_path = exp_dir / 'training_history.yaml'
        
        if not history_path.exists():
            self._log(f"Warning: No training history found for {exp_name}")
            return
        
        with open(history_path, 'r') as f:
            history = yaml.safe_load(f)
        
        # Store results
        result = {
            'exp_name': exp_name,
            'best_miou': float(max(history['val_miou'])),
            'final_miou': float(history['val_miou'][-1]),
            'best_dice': float(max(history['val_dice'])),
            'final_dice': float(history['val_dice'][-1]),
            'best_pixel_acc': float(max(history['val_pixel_acc'])),
            'final_pixel_acc': float(history['val_pixel_acc'][-1]),
            'min_train_loss': float(min(history['train_loss'])),
            'final_train_loss': float(history['train_loss'][-1]),
            'num_epochs': len(history['train_loss']),
        }
        
        # Add config-specific fields
        result.update(config_overrides)
        self.results.append(result)
        
        self._log(f"Results: mIoU={result['best_miou']:.4f}, Dice={result['best_dice']:.4f}")
    
    def run_experiment_1(self):
        """
        Experiment 1: Effect of Number of Point Annotations
        Tests: [1, 3, 5, 10, 20, 50] points per class + Full supervision
        """
        self._log("\n" + "="*70)
        self._log("EXPERIMENT 1: Effect of Number of Point Annotations")
        self._log("="*70)
        
        # Define point configurations to test
        num_points_list = [1, 3, 5, 10, 20, 50]
        
        # Run point supervision experiments
        for num_points in num_points_list:
            exp_name = f"exp1_points_{num_points}"
            config_overrides = {
                'training.num_points_per_class': num_points,
                'training.sampling_strategy': 'random',
                'training.use_point_supervision': True,
                'experiment.points': num_points,
                'experiment.type': 'num_points',
            }
            
            self.run_experiment(exp_name, config_overrides)
        
        # Run full supervision baseline
        exp_name = "exp1_full_supervision"
        config_overrides = {
            'training.use_point_supervision': False,
            'experiment.points': 'Full',
            'experiment.type': 'num_points',
        }
        
        self.run_experiment(exp_name, config_overrides)
        
        # Generate experiment 1 results
        self._generate_exp1_results()
    
    def run_experiment_2(self):
        """
        Experiment 2: Comparison of Sampling Strategies
        Tests: random, centroid, boundary, grid
        """
        self._log("\n" + "="*70)
        self._log("EXPERIMENT 2: Comparison of Sampling Strategies")
        self._log("="*70)
        
        strategies = ['random', 'centroid', 'boundary', 'grid']
        num_points = 10  # Fixed number of points for fair comparison
        
        for strategy in strategies:
            exp_name = f"exp2_strategy_{strategy}"
            config_overrides = {
                'training.sampling_strategy': strategy,
                'training.num_points_per_class': num_points,
                'training.use_point_supervision': True,
                'experiment.strategy': strategy,
                'experiment.type': 'sampling_strategy',
            }
            
            self.run_experiment(exp_name, config_overrides)
        
        # Generate experiment 2 results
        self._generate_exp2_results()
    
    def _generate_exp1_results(self):
        """Generate publication-ready results for Experiment 1"""
        self._log("\n" + "="*70)
        self._log("Generating Experiment 1 Results...")
        self._log("="*70)
        
        # Filter exp1 results
        exp1_results = [r for r in self.results if r.get('experiment.type') == 'num_points']
        
        if not exp1_results:
            self._log("No Experiment 1 results found!")
            return
        
        # Create DataFrame
        df = pd.DataFrame(exp1_results)
        
        # Sort by number of points
        df_numeric = df[df['experiment.points'] != 'Full'].copy()
        df_numeric['experiment.points'] = df_numeric['experiment.points'].astype(int)
        df_numeric = df_numeric.sort_values('experiment.points')
        
        df_full = df[df['experiment.points'] == 'Full']
        
        # Save detailed results table
        table_path = self.tables_dir / 'exp1_detailed_results.csv'
        df.to_csv(table_path, index=False)
        self._log(f"Detailed results saved: {table_path}")
        
        # Create summary table
        summary_cols = ['experiment.points', 'best_miou', 'best_dice', 'best_pixel_acc', 'num_epochs']
        summary_df = df[summary_cols].copy()
        summary_df.columns = ['Points per Class', 'Best mIoU', 'Best Dice', 'Best Pixel Acc', 'Epochs']
        summary_df = summary_df.round(4)
        
        # Save as CSV and formatted text
        summary_path = self.tables_dir / 'exp1_summary_table.csv'
        summary_df.to_csv(summary_path, index=False)
        
        # Save formatted text table for report
        table_text_path = self.tables_dir / 'exp1_summary_table.txt'
        with open(table_text_path, 'w') as f:
            f.write("Table 1: Effect of Number of Point Annotations\n")
            f.write("="*80 + "\n\n")
            f.write(summary_df.to_string(index=False))
            f.write("\n\n")
        
        self._log(f"Summary table saved: {summary_path}")
        
        # Generate plots
        self._plot_exp1_results(df_numeric, df_full)
    
    def _plot_exp1_results(self, df_numeric, df_full):
        """Create publication-ready plots for Experiment 1"""
        
        # Figure 1: mIoU vs Number of Points (with full supervision baseline)
        fig, ax = plt.subplots(figsize=(10, 6))
        
        x = df_numeric['experiment.points'].values
        y_miou = df_numeric['best_miou'].values
        
        # Plot point supervision results
        ax.plot(x, y_miou, 'o-', linewidth=2.5, markersize=10, 
                label='Point Supervision', color='#2E86AB', zorder=3)
        
        # Add full supervision baseline
        if not df_full.empty:
            full_miou = df_full['best_miou'].values[0]
            ax.axhline(y=full_miou, color='#A23B72', linestyle='--', 
                      linewidth=2.5, label='Full Supervision', zorder=2)
            
            # Add text annotation
            ax.text(x[-1], full_miou + 0.01, f'Full: {full_miou:.3f}', 
                   fontsize=11, color='#A23B72', fontweight='bold')
        
        ax.set_xlabel('Number of Points per Class', fontsize=14, fontweight='bold')
        ax.set_ylabel('Mean IoU (mIoU)', fontsize=14, fontweight='bold')
        ax.set_title('Experiment 1: Effect of Number of Point Annotations on Performance', 
                    fontsize=16, fontweight='bold', pad=20)
        ax.legend(fontsize=12, loc='lower right', framealpha=0.9)
        ax.grid(True, alpha=0.3, linestyle='--')
        ax.set_ylim(bottom=0)
        
        # Add value labels on points
        for xi, yi in zip(x, y_miou):
            ax.annotate(f'{yi:.3f}', (xi, yi), textcoords="offset points", 
                       xytext=(0,10), ha='center', fontsize=9)
        
        plt.tight_layout()
        fig_path = self.figures_dir / 'exp1_miou_vs_points.png'
        plt.savefig(fig_path, dpi=300, bbox_inches='tight')
        plt.close()
        self._log(f"Figure saved: {fig_path}")
        
        # Figure 2: Multiple metrics comparison
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
        
        metrics = [
            ('best_miou', 'mIoU', '#2E86AB'),
            ('best_dice', 'Dice Score', '#F18F01'),
            ('best_pixel_acc', 'Pixel Accuracy', '#C73E1D')
        ]
        
        for idx, (metric, label, color) in enumerate(metrics):
            y = df_numeric[metric].values
            axes[idx].plot(x, y, 'o-', linewidth=2.5, markersize=10, 
                          color=color, label=label)
            
            if not df_full.empty:
                full_value = df_full[metric].values[0]
                axes[idx].axhline(y=full_value, color=color, linestyle='--', 
                                 linewidth=2, alpha=0.7, label='Full Supervision')
            
            axes[idx].set_xlabel('Points per Class', fontsize=12, fontweight='bold')
            axes[idx].set_ylabel(label, fontsize=12, fontweight='bold')
            axes[idx].set_title(f'{label} vs Points', fontsize=14, fontweight='bold')
            axes[idx].legend(fontsize=10)
            axes[idx].grid(True, alpha=0.3)
            axes[idx].set_ylim(bottom=0)
        
        plt.tight_layout()
        fig_path = self.figures_dir / 'exp1_all_metrics.png'
        plt.savefig(fig_path, dpi=300, bbox_inches='tight')
        plt.close()
        self._log(f"Figure saved: {fig_path}")
        
        # Figure 3: Performance gain analysis
        fig, ax = plt.subplots(figsize=(10, 6))
        
        if not df_full.empty:
            full_miou = df_full['best_miou'].values[0]
            relative_performance = (y_miou / full_miou) * 100
            
            ax.bar(x, relative_performance, color='#2E86AB', alpha=0.7, edgecolor='black', linewidth=1.5)
            ax.axhline(y=100, color='#A23B72', linestyle='--', linewidth=2.5, 
                      label='Full Supervision (100%)')
            
            ax.set_xlabel('Number of Points per Class', fontsize=14, fontweight='bold')
            ax.set_ylabel('Relative Performance (%)', fontsize=14, fontweight='bold')
            ax.set_title('Experiment 1: Performance Relative to Full Supervision', 
                        fontsize=16, fontweight='bold', pad=20)
            ax.legend(fontsize=12)
            ax.grid(True, alpha=0.3, axis='y')
            ax.set_ylim(0, 110)
            
            # Add percentage labels
            for xi, yi in zip(x, relative_performance):
                ax.text(xi, yi + 2, f'{yi:.1f}%', ha='center', fontsize=10, fontweight='bold')
            
            plt.tight_layout()
            fig_path = self.figures_dir / 'exp1_relative_performance.png'
            plt.savefig(fig_path, dpi=300, bbox_inches='tight')
            plt.close()
            self._log(f"Figure saved: {fig_path}")
    
    def _generate_exp2_results(self):
        """Generate publication-ready results for Experiment 2"""
        self._log("\n" + "="*70)
        self._log("Generating Experiment 2 Results...")
        self._log("="*70)
        
        # Filter exp2 results
        exp2_results = [r for r in self.results if r.get('experiment.type') == 'sampling_strategy']
        
        if not exp2_results:
            self._log("No Experiment 2 results found!")
            return
        
        # Create DataFrame
        df = pd.DataFrame(exp2_results)
        df = df.sort_values('best_miou', ascending=False)
        
        # Save detailed results
        table_path = self.tables_dir / 'exp2_detailed_results.csv'
        df.to_csv(table_path, index=False)
        self._log(f"Detailed results saved: {table_path}")
        
        # Create summary table
        summary_cols = ['experiment.strategy', 'best_miou', 'best_dice', 'best_pixel_acc']
        summary_df = df[summary_cols].copy()
        summary_df.columns = ['Sampling Strategy', 'Best mIoU', 'Best Dice', 'Best Pixel Acc']
        summary_df = summary_df.round(4)
        
        summary_path = self.tables_dir / 'exp2_summary_table.csv'
        summary_df.to_csv(summary_path, index=False)
        
        # Save formatted text table
        table_text_path = self.tables_dir / 'exp2_summary_table.txt'
        with open(table_text_path, 'w') as f:
            f.write("Table 2: Comparison of Point Sampling Strategies\n")
            f.write("="*80 + "\n\n")
            f.write(summary_df.to_string(index=False))
            f.write("\n\n")
        
        self._log(f"Summary table saved: {summary_path}")
        
        # Generate plots
        self._plot_exp2_results(df)
    
    def _plot_exp2_results(self, df):
        """Create publication-ready plots for Experiment 2"""
        
        strategies = df['experiment.strategy'].values
        
        # Figure 1: Bar chart comparison of strategies
        fig, ax = plt.subplots(figsize=(10, 6))
        
        colors = ['#2E86AB', '#F18F01', '#C73E1D', '#6A994E']
        bars = ax.bar(strategies, df['best_miou'].values, color=colors, 
                     alpha=0.8, edgecolor='black', linewidth=1.5)
        
        ax.set_xlabel('Sampling Strategy', fontsize=14, fontweight='bold')
        ax.set_ylabel('Mean IoU (mIoU)', fontsize=14, fontweight='bold')
        ax.set_title('Experiment 2: Comparison of Point Sampling Strategies', 
                    fontsize=16, fontweight='bold', pad=20)
        ax.grid(True, alpha=0.3, axis='y')
        ax.set_ylim(bottom=0)
        
        # Add value labels on bars
        for bar, value in zip(bars, df['best_miou'].values):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                   f'{value:.4f}', ha='center', va='bottom', 
                   fontsize=11, fontweight='bold')
        
        plt.tight_layout()
        fig_path = self.figures_dir / 'exp2_strategy_comparison.png'
        plt.savefig(fig_path, dpi=300, bbox_inches='tight')
        plt.close()
        self._log(f"Figure saved: {fig_path}")
        
        # Figure 2: Multiple metrics grouped bar chart
        fig, ax = plt.subplots(figsize=(12, 6))
        
        metrics = ['best_miou', 'best_dice', 'best_pixel_acc']
        metric_labels = ['mIoU', 'Dice', 'Pixel Acc']
        
        x = np.arange(len(strategies))
        width = 0.25
        
        for i, (metric, label) in enumerate(zip(metrics, metric_labels)):
            offset = (i - 1) * width
            bars = ax.bar(x + offset, df[metric].values, width, 
                         label=label, alpha=0.8, edgecolor='black', linewidth=1)
        
        ax.set_xlabel('Sampling Strategy', fontsize=14, fontweight='bold')
        ax.set_ylabel('Score', fontsize=14, fontweight='bold')
        ax.set_title('Experiment 2: Multi-Metric Comparison of Sampling Strategies', 
                    fontsize=16, fontweight='bold', pad=20)
        ax.set_xticks(x)
        ax.set_xticklabels(strategies)
        ax.legend(fontsize=12, loc='lower right')
        ax.grid(True, alpha=0.3, axis='y')
        ax.set_ylim(bottom=0)
        
        plt.tight_layout()
        fig_path = self.figures_dir / 'exp2_multi_metric_comparison.png'
        plt.savefig(fig_path, dpi=300, bbox_inches='tight')
        plt.close()
        self._log(f"Figure saved: {fig_path}")
        
        # Figure 3: Performance ranking
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Calculate relative performance (compared to best)
        best_miou = df['best_miou'].max()
        relative = (df['best_miou'].values / best_miou) * 100
        
        bars = ax.barh(strategies, relative, color=colors, alpha=0.8, 
                      edgecolor='black', linewidth=1.5)
        
        ax.set_xlabel('Relative Performance (%)', fontsize=14, fontweight='bold')
        ax.set_ylabel('Sampling Strategy', fontsize=14, fontweight='bold')
        ax.set_title('Experiment 2: Relative Performance Ranking', 
                    fontsize=16, fontweight='bold', pad=20)
        ax.grid(True, alpha=0.3, axis='x')
        ax.set_xlim(90, 105)
        
        # Add percentage labels
        for bar, value in zip(bars, relative):
            width = bar.get_width()
            ax.text(width + 0.5, bar.get_y() + bar.get_height()/2.,
                   f'{value:.1f}%', ha='left', va='center', 
                   fontsize=11, fontweight='bold')
        
        plt.tight_layout()
        fig_path = self.figures_dir / 'exp2_ranking.png'
        plt.savefig(fig_path, dpi=300, bbox_inches='tight')
        plt.close()
        self._log(f"Figure saved: {fig_path}")
    
    def generate_final_report_summary(self):
        """Generate a comprehensive summary for report writing"""
        self._log("\n" + "="*70)
        self._log("Generating Final Report Summary...")
        self._log("="*70)
        
        summary_path = self.results_dir / 'REPORT_SUMMARY.txt'
        
        with open(summary_path, 'w') as f:
            f.write("="*80 + "\n")
            f.write("COMPREHENSIVE EXPERIMENT RESULTS SUMMARY\n")
            f.write("For Technical Report Writing\n")
            f.write("="*80 + "\n\n")
            
            # Experiment 1 Summary
            exp1_results = [r for r in self.results if r.get('experiment.type') == 'num_points']
            if exp1_results:
                f.write("EXPERIMENT 1: Effect of Number of Point Annotations\n")
                f.write("-"*80 + "\n\n")
                
                df1 = pd.DataFrame(exp1_results)
                df1_numeric = df1[df1['experiment.points'] != 'Full'].copy()
                df1_full = df1[df1['experiment.points'] == 'Full']
                
                f.write(f"Number of configurations tested: {len(df1_numeric)}\n")
                f.write(f"Point configurations: {sorted(df1_numeric['experiment.points'].tolist())}\n\n")
                
                f.write("Key Findings:\n")
                best_point = df1_numeric.loc[df1_numeric['best_miou'].idxmax()]
                f.write(f"  - Best point configuration: {best_point['experiment.points']} points\n")
                f.write(f"  - Best mIoU: {best_point['best_miou']:.4f}\n")
                
                if not df1_full.empty:
                    full_miou = df1_full['best_miou'].values[0]
                    f.write(f"  - Full supervision mIoU: {full_miou:.4f}\n")
                    f.write(f"  - Best point config achieves {(best_point['best_miou']/full_miou)*100:.1f}% of full supervision\n")
                
                f.write("\n")
            
            # Experiment 2 Summary
            exp2_results = [r for r in self.results if r.get('experiment.type') == 'sampling_strategy']
            if exp2_results:
                f.write("EXPERIMENT 2: Comparison of Sampling Strategies\n")
                f.write("-"*80 + "\n\n")
                
                df2 = pd.DataFrame(exp2_results)
                f.write(f"Number of strategies tested: {len(df2)}\n")
                f.write(f"Strategies: {df2['experiment.strategy'].tolist()}\n\n")
                
                f.write("Key Findings:\n")
                best_strategy = df2.loc[df2['best_miou'].idxmax()]
                worst_strategy = df2.loc[df2['best_miou'].idxmin()]
                
                f.write(f"  - Best strategy: {best_strategy['experiment.strategy']}\n")
                f.write(f"  - Best mIoU: {best_strategy['best_miou']:.4f}\n")
                f.write(f"  - Worst strategy: {worst_strategy['experiment.strategy']}\n")
                f.write(f"  - Worst mIoU: {worst_strategy['best_miou']:.4f}\n")
                f.write(f"  - Performance gap: {(best_strategy['best_miou'] - worst_strategy['best_miou']):.4f}\n")
                
                f.write("\n")
            
            f.write("\n" + "="*80 + "\n")
            f.write("FILES GENERATED FOR REPORT\n")
            f.write("="*80 + "\n\n")
            
            f.write("Figures (in figures/ directory):\n")
            for fig_file in sorted(self.figures_dir.glob('*.png')):
                f.write(f"  - {fig_file.name}\n")
            
            f.write("\nTables (in tables/ directory):\n")
            for table_file in sorted(self.tables_dir.glob('*.csv')):
                f.write(f"  - {table_file.name}\n")
            
            f.write("\nFormatted Tables (in tables/ directory):\n")
            for table_file in sorted(self.tables_dir.glob('*.txt')):
                f.write(f"  - {table_file.name}\n")
            
            f.write("\n")
        
        self._log(f"Report summary saved: {summary_path}")
        
        # Save all results as single JSON
        json_path = self.results_dir / 'all_results.json'
        with open(json_path, 'w') as f:
            json.dump(self.results, f, indent=2)
        self._log(f"All results JSON saved: {json_path}")


def main(args):
    """Main function"""
    
    print("\n" + "="*70)
    print("ENHANCED AUTOMATED EXPERIMENT RUNNER")
    print("="*70)
    print(f"Experiment: {args.experiment}")
    print(f"Base config: {args.config}")
    print(f"Output directory: {args.output_dir}")
    print("="*70 + "\n")
    
    # Create runner
    runner = EnhancedExperimentRunner(
        base_config_path=args.config,
        output_base_dir=args.output_dir
    )
    
    # Run experiments
    if args.experiment in ['exp1', 'all']:
        runner.run_experiment_1()
    
    if args.experiment in ['exp2', 'all']:
        runner.run_experiment_2()
    
    # Generate final summary
    runner.generate_final_report_summary()
    
    print("\n" + "="*70)
    print("ALL EXPERIMENTS COMPLETE!")
    print("="*70)
    print(f"\nResults saved to: {args.output_dir}")
    print(f"\nOrganized structure:")
    print(f"  - Figures: {runner.figures_dir}")
    print(f"  - Tables: {runner.tables_dir}")
    print(f"  - Checkpoints: {runner.checkpoints_dir}")
    print(f"  - Summary: {runner.results_dir}/REPORT_SUMMARY.txt")
    print("\n" + "="*70 + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Enhanced automated experiments')
    parser.add_argument(
        '--experiment',
        type=str,
        choices=['exp1', 'exp2', 'all'],
        default='all',
        help='Which experiment to run'
    )
    parser.add_argument(
        '--config',
        type=str,
        default='config/default.yaml',
        help='Base configuration file'
    )
    parser.add_argument(
        '--output_dir',
        type=str,
        default='./experiments_results',
        help='Output directory for all results'
    )
    
    args = parser.parse_args()
    main(args)