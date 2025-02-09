import os
import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

def plot_model_metrics():
    """Plot model performance metrics comparison."""
    today = datetime.now().strftime("%Y%m%d")
    stats = ['PTS', 'TRB', 'AST', '3P']
    metrics_data = []
    
    for stat in stats:
        with open(f"src/models/{stat}_metrics_{today}.json", 'r') as f:
            metrics = json.load(f)
            metrics_data.append({
                'Statistic': stat,
                'R²': metrics['r2'],
                'RMSE': metrics['rmse'],
                'MAE': metrics['mae']
            })
    
    metrics_df = pd.DataFrame(metrics_data)
    
    # Plot R² scores
    plt.figure(figsize=(10, 6))
    sns.barplot(data=metrics_df, x='Statistic', y='R²')
    plt.title('Model Performance Comparison (R² Score)')
    plt.ylim(0, 1)  # R² is between 0 and 1
    for i, v in enumerate(metrics_df['R²']):
        plt.text(i, v + 0.01, f'{v:.3f}', ha='center')
    plt.tight_layout()
    plt.savefig('src/models/model_r2_comparison.png')
    plt.close()
    
    # Plot Error Metrics
    plt.figure(figsize=(12, 6))
    metrics_long = pd.melt(metrics_df, 
                          id_vars=['Statistic'], 
                          value_vars=['RMSE', 'MAE'],
                          var_name='Metric',
                          value_name='Value')
    
    sns.barplot(data=metrics_long, x='Statistic', y='Value', hue='Metric')
    plt.title('Model Error Metrics Comparison')
    for i, v in enumerate(metrics_long['Value']):
        plt.text(i // 2, v, f'{v:.2f}', ha='center')
    plt.tight_layout()
    plt.savefig('src/models/model_error_comparison.png')
    plt.close()

def main():
    """Generate model metrics visualizations."""
    os.makedirs('src/models', exist_ok=True)
    plot_model_metrics()
    print("Model performance visualizations have been generated in the models directory!")

if __name__ == "__main__":
    main()
