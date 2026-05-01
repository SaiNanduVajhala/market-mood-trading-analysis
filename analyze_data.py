import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

# set up some basic plotting styles
sns.set_theme(style="whitegrid")

def main():
    print("loading up the datasets...")
    
    try:
        sentiment_df = pd.read_csv('fear_greed_index.csv')
        trades_df = pd.read_csv('historical_data.csv')
    except FileNotFoundError:
        print("Couldn't find the CSV files. Make sure they are in the same folder.")
        return

    print("cleaning dates to get them ready for a merge...")
    
    # fix up the dates in the sentiment data
    sentiment_df['date'] = pd.to_datetime(sentiment_df['date'])
    sentiment_df['Date'] = sentiment_df['date'].dt.date
    sentiment_df = sentiment_df[['Date', 'value', 'classification']].rename(
        columns={'value': 'fg_value', 'classification': 'sentiment'}
    )

    # fix up dates in the trading data
    # the timestamp is usually something like '02-12-2024 22:50'
    trades_df['Timestamp IST'] = pd.to_datetime(trades_df['Timestamp IST'], format='%d-%m-%Y %H:%M')
    trades_df['Date'] = trades_df['Timestamp IST'].dt.date
    
    # factor in fees to get the actual net pnl for each trade
    trades_df['net_pnl'] = trades_df['Closed PnL'] - trades_df['Fee']
    trades_df['is_win'] = trades_df['net_pnl'] > 0
    
    print("merging trade history with daily market sentiment...")
    merged_df = pd.merge(trades_df, sentiment_df, on='Date', how='left')
    
    # drop rows where we don't have sentiment data (probably weekends or missing days)
    missing_sentiment = merged_df['sentiment'].isnull().sum()
    if missing_sentiment > 0:
        print(f"heads up: dropping {missing_sentiment} trades because they lack sentiment data")
        merged_df = merged_df.dropna(subset=['sentiment'])
    
    # Let's see how people perform across different market moods
    print("\n--- overall performance by sentiment ---")
    perf_summary = merged_df.groupby('sentiment').agg(
        total_trades=('Trade ID', 'count'),
        win_rate=('is_win', 'mean'),
        net_pnl=('net_pnl', 'sum'),
        total_volume=('Size USD', 'sum')
    ).reset_index()
    
    print(perf_summary)
    
    # make sure we have a folder for the plots
    if not os.path.exists('plots'):
        os.makedirs('plots')
    
    # order the sentiments logically for the charts
    sentiment_order = ['Extreme Fear', 'Fear', 'Neutral', 'Greed', 'Extreme Greed']
    
    # Plot 1: How much money was actually made/lost?
    plt.figure(figsize=(10, 6))
    sns.barplot(x='sentiment', y='net_pnl', data=perf_summary, order=sentiment_order, palette='viridis')
    plt.title('Net PnL Across Market Sentiments')
    plt.ylabel('Net PnL ($)')
    plt.xlabel('Market Mood')
    plt.tight_layout()
    plt.savefig('plots/pnl_by_sentiment.png')
    plt.close()
    
    # Plot 2: What about the win rate?
    plt.figure(figsize=(10, 6))
    sns.barplot(x='sentiment', y='win_rate', data=perf_summary, order=sentiment_order, palette='mako')
    plt.title('Win Rate Across Market Sentiments')
    plt.ylabel('Win Rate (%)')
    plt.xlabel('Market Mood')
    plt.ylim(0, 1)
    plt.tight_layout()
    plt.savefig('plots/winrate_by_sentiment.png')
    plt.close()
    
    # Let's break it down by Long vs Short (buy vs sell) to see if there's an edge
    print("\n--- long vs short breakdown ---")
    side_perf = merged_df.groupby(['sentiment', 'Side']).agg(
        trade_count=('Trade ID', 'count'),
        volume=('Size USD', 'sum'),
        net_pnl=('net_pnl', 'sum')
    ).reset_index()
    
    print(side_perf)
    
    # Plot 3: Buying vs Selling volumes
    plt.figure(figsize=(12, 6))
    sns.barplot(x='sentiment', y='volume', hue='Side', data=side_perf, order=sentiment_order, palette='coolwarm')
    plt.title('Trading Volume (Buy vs Sell) by Sentiment')
    plt.ylabel('Total Volume ($)')
    plt.xlabel('Market Mood')
    plt.tight_layout()
    plt.savefig('plots/volume_by_side.png')
    plt.close()

    # save out the summary table so we have the raw numbers
    perf_summary.to_csv('plots/sentiment_summary.csv', index=False)
    print("\nall done! check the 'plots' folder for the charts and summary data.")

if __name__ == "__main__":
    main()
