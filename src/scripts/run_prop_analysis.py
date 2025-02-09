import pandas as pd
from odds.odds_api import OddsAPI
from analysis.prop_analyzer import PropAnalyzer

def main():
    # Load player features
    features_df = pd.read_csv("src/data/features/nba_player_stats_features_20250208.csv", index_col="Player")
    
    # Convert boolean columns to integers
    bool_cols = features_df.select_dtypes(include=['bool']).columns
    for col in bool_cols:
        features_df[col] = features_df[col].astype(int)
    
    # Convert integer columns to float
    int_cols = features_df.select_dtypes(include=['int64']).columns
    for col in int_cols:
        features_df[col] = features_df[col].astype(float)
    
    # Initialize APIs and analyzer
    odds_api = OddsAPI()
    analyzer = PropAnalyzer()
    
    # Fetch all available props
    print("Fetching props from OddsAPI...")
    props = odds_api.get_all_props()
    odds_api.save_props(props)
    
    # Analyze props
    print("\nAnalyzing props...")
    analyzed_props = analyzer.analyze_props(props, features_df)
    
    # Find best edges
    print("\nFinding best edges...")
    best_props = analyzer.find_best_edges(analyzed_props, min_edge=0.05)
    
    # Save analysis
    analyzer.save_analysis(best_props)
    
    # Print results
    print("\nTop 10 Props with Best Edges:")
    print("-" * 100)
    for prop in best_props[:10]:
        print(
            f"{prop['player_name']} - {prop['market']} - "
            f"Line: {prop['line']}, Prediction: {prop['prediction']:.1f}, "
            f"Price: {prop['price']:+d}, Edge: {prop['edge']:.1f}%, "
            f"Kelly: {prop['kelly']*100:.1f}%, Score: {prop['score']:.1f}"
        )
    
    # Print summary statistics
    total_props = len(analyzed_props)
    props_with_edge = len([p for p in analyzed_props if abs(p['edge']) >= 5.0])
    avg_edge = sum(abs(p['edge']) for p in analyzed_props) / total_props
    
    print(f"\nSummary:")
    print(f"Total props analyzed: {total_props}")
    print(f"Props with edge >= 5%: {props_with_edge} ({props_with_edge/total_props*100:.1f}%)")
    print(f"Average absolute edge: {avg_edge:.1f}%")

if __name__ == "__main__":
    main()
