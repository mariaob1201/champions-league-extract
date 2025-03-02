import pandas as pd
import re
from collections import Counter
import matplotlib.pyplot as plt
import seaborn as sns
import nltk
nltk.download('vader_lexicon')
from nltk.sentiment.vader import SentimentIntensityAnalyzer


# Initialize Sentiment Analyzer
sia = SentimentIntensityAnalyzer()

# Load the dataset containing posts
posts_df = pd.read_csv('champions_league_posts.csv')

# Define a list of Real Madrid players (update this list with actual current players)
real_madrid_players = [
    'Courtois', 'Alaba', 'Carvajal', 'Militao', 'Rüdiger', 'Nacho', 'Modric',
    'Kroos', 'Casemiro', 'Benzema', 'Vinicius', 'Rodrygo', 'Valverde', 'Hazard',
    'Asensio', 'Mendy', 'Ceballos', 'Camavinga', 'Brahim', 'Isco', 'Jovic', 'Tchouaméni'
]

# Convert player names to a pattern to match in the text (case insensitive)
player_patterns = [re.compile(r'\b' + re.escape(player) + r'\b', re.IGNORECASE) for player in real_madrid_players]

# Function to count player mentions in text (title and selftext)
def count_player_mentions_and_sentiment(text):
    mentions = []
    sentiments = []
    if pd.isna(text):
        return mentions, sentiments
    # Check for mentions of players
    for pattern in player_patterns:
        if pattern.search(text):
            player_name = pattern.pattern[2:-2]  # Extract player name from pattern
            mentions.append(player_name)
            sentiment_score = sia.polarity_scores(text)  # Get sentiment scores for the text
            sentiment = 'positive' if sentiment_score['compound'] > 0 else 'negative' if sentiment_score['compound'] < 0 else 'neutral'
            sentiments.append((player_name, sentiment))
    return mentions, sentiments

# Apply the function to both 'title' and 'selftext' columns to extract mentions and their sentiments
posts_df['player_mentions'], posts_df['player_sentiments'] = zip(*posts_df.apply(
    lambda row: count_player_mentions_and_sentiment(str(row['title']) + " " + str(row['selftext'])), axis=1))

# Flatten the list of all mentioned players and their sentiments
all_player_mentions = [player for sublist in posts_df['player_mentions'] for player in sublist]
all_player_sentiments = [sentiment for sublist in posts_df['player_sentiments'] for sentiment in sublist]

# Create a DataFrame from the flattened lists
sentiment_df = pd.DataFrame(all_player_sentiments, columns=["player", "sentiment"])

# Count the frequency of each sentiment for each player
sentiment_counts_df = sentiment_df.groupby(['player', 'sentiment']).size().unstack(fill_value=0)

# Ensure 'positive', 'negative', and 'neutral' columns are present
for sentiment in ['positive', 'negative', 'neutral']:
    if sentiment not in sentiment_counts_df.columns:
        sentiment_counts_df[sentiment] = 0

# Calculate total mentions for each player
sentiment_counts_df['total_mentions'] = sentiment_counts_df.sum(axis=1)

# Calculate relative metrics (proportions of positive, negative, and neutral mentions)
sentiment_counts_df['positive_percentage'] = sentiment_counts_df['positive'] / sentiment_counts_df['total_mentions'] * 100
sentiment_counts_df['negative_percentage'] = sentiment_counts_df['negative'] / sentiment_counts_df['total_mentions'] * 100
sentiment_counts_df['neutral_percentage'] = sentiment_counts_df['neutral'] / sentiment_counts_df['total_mentions'] * 100

# Visualize the relative sentiment percentages for top 10 most mentioned players
top_players = sentiment_counts_df['total_mentions'].nlargest(10).index
top_sentiment_counts_df = sentiment_counts_df.loc[top_players, ['positive_percentage', 'negative_percentage', 'neutral_percentage']]

# Plotting the relative sentiment percentages
top_sentiment_counts_df.plot(kind='barh', stacked=True, figsize=(10, 6), cmap="coolwarm")

plt.title("Relative Positive vs Negative Mentions of Real Madrid Players")
plt.xlabel("Percentage of Total Mentions")
plt.ylabel("Players")
plt.show()