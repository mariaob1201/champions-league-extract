import pandas as pd
import re
from collections import Counter
import matplotlib.pyplot as plt
import seaborn as sns
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from nltk.tokenize import sent_tokenize
import numpy as np
from wordcloud import WordCloud
from sklearn.feature_extraction.text import CountVectorizer
import spacy
import warnings
import nltk
nltk.download('punkt')

from nltk.sentiment.vader import SentimentIntensityAnalyzer
from nltk.tokenize import PunktSentenceTokenizer

tokenizer = PunktSentenceTokenizer()




warnings.filterwarnings('ignore')

# Download necessary NLTK packages
try:
    nltk.download('vader_lexicon', quiet=True)
    nltk.download('punkt', quiet=True)
except:
    print("NLTK download failed, but continuing with execution")

# Load spaCy for better entity recognition
try:
    nlp = spacy.load("en_core_web_sm")
    using_spacy = True
except:
    print("spaCy model not available. Falling back to regex patterns.")
    using_spacy = False

# Initialize Sentiment Analyzer
sia = SentimentIntensityAnalyzer()

# Load the dataset containing posts
posts_df = pd.read_csv('datasets/champions_league_posts.csv')

# Make sure to handle NaN values
posts_df['selftext'] = posts_df['selftext'].fillna('')

# Define a comprehensive list of Real Madrid players with variations
real_madrid_players = {
    'Courtois': ['Courtois', 'Thibaut', 'Thibaut Courtois'],
    'Lunin': ['Lunin', 'Andriy', 'Andriy Lunin'],
    'Alaba': ['Alaba', 'David', 'David Alaba'],
    'Carvajal': ['Carvajal', 'Dani', 'Daniel Carvajal'],
    'Militao': ['Militao', 'Éder', 'Eder Militao', 'Éder Militão'],
    'Rüdiger': ['Rüdiger', 'Rudiger', 'Antonio', 'Antonio Rudiger', 'Antonio Rüdiger'],
    'Nacho': ['Nacho', 'Nacho Fernandez', 'Fernandez'],
    'Mendy': ['Mendy', 'Ferland', 'Ferland Mendy'],
    'Fran García': ['Fran García', 'Fran Garcia', 'García'],
    'Lucas Vázquez': ['Lucas Vázquez', 'Lucas Vazquez', 'Vázquez', 'Vazquez'],
    'Modric': ['Modric', 'Luka', 'Luka Modric', 'Modrić'],
    'Kroos': ['Kroos', 'Toni', 'Toni Kroos'],
    'Valverde': ['Valverde', 'Fede', 'Federico', 'Federico Valverde'],
    'Tchouaméni': ['Tchouaméni', 'Tchouameni', 'Aurélien', 'Aurelien', 'Aurélien Tchouaméni'],
    'Camavinga': ['Camavinga', 'Eduardo', 'Eduardo Camavinga'],
    'Bellingham': ['Bellingham', 'Jude', 'Jude Bellingham'],
    'Ceballos': ['Ceballos', 'Dani', 'Daniel Ceballos'],
    'Güler': ['Güler', 'Guler', 'Arda', 'Arda Güler', 'Arda Guler'],
    'Benzema': ['Benzema', 'Karim', 'Karim Benzema'],
    'Vinicius': ['Vinicius', 'Vini', 'Vinicius Jr', 'Vinicius Junior'],
    'Rodrygo': ['Rodrygo', 'Rodrygo Goes'],
    'Joselu': ['Joselu'],
    'Brahim': ['Brahim', 'Brahim Díaz', 'Brahim Diaz', 'Díaz'],
    'Endrick': ['Endrick', 'Endrick Felipe'],
    'Mbappé': ['Mbappé', 'Mbappe', 'Kylian', 'Kylian Mbappé', 'Kylian Mbappe'],
    'Ancelotti': ['Ancelotti', 'Carlo', 'Carlo Ancelotti', 'Don Carlo']  # Including the coach
}

# Flatten the dictionary to create a list of all variations
all_variations = []
player_to_canonical = {}
for canonical, variations in real_madrid_players.items():
    all_variations.extend(variations)
    for var in variations:
        player_to_canonical[var.lower()] = canonical

# Convert player name variations to a pattern to match in the text (case insensitive)
player_patterns = {re.compile(r'\b' + re.escape(variation) + r'\b', re.IGNORECASE): player
                   for player, variations in real_madrid_players.items()
                   for variation in variations}


# Function to extract player mentions with context from text
def extract_player_mentions_with_context(text, window_size=50):
    if pd.isna(text) or text == '':
        return []

    mentions = []

    tokenizer = PunktSentenceTokenizer()
    sentences = tokenizer.tokenize(text)

    # Using spaCy for entity recognition if available
    if using_spacy:
        doc = nlp(text)
        for ent in doc.ents:
            if ent.text.lower() in player_to_canonical:
                canonical_name = player_to_canonical[ent.text.lower()]
                start = max(0, ent.start_char - window_size)
                end = min(len(text), ent.end_char + window_size)
                context = text[start:end]
                sentiment = sia.polarity_scores(context)
                mentions.append({
                    'player': canonical_name,
                    'context': context,
                    'sentiment_compound': sentiment['compound'],
                    'sentiment': 'positive' if sentiment['compound'] > 0.05 else 'negative' if sentiment[
                                                                                                   'compound'] < -0.05 else 'neutral'
                })

        # If spaCy didn't find any players, fall back to regex
        if not mentions:
            for pattern, player in player_patterns.items():
                for match in pattern.finditer(text):
                    start = max(0, match.start() - window_size)
                    end = min(len(text), match.end() + window_size)
                    context = text[start:end]
                    sentiment = sia.polarity_scores(context)
                    mentions.append({
                        'player': player,
                        'context': context,
                        'sentiment_compound': sentiment['compound'],
                        'sentiment': 'positive' if sentiment['compound'] > 0.05 else 'negative' if sentiment[
                                                                                                       'compound'] < -0.05 else 'neutral'
                    })
    else:
        # Regex-based approach
        for sentence in sentences:
            for pattern, player in player_patterns.items():
                if pattern.search(sentence):
                    sentiment = sia.polarity_scores(sentence)
                    mentions.append({
                        'player': player,
                        'context': sentence,
                        'sentiment_compound': sentiment['compound'],
                        'sentiment': 'positive' if sentiment['compound'] > 0.05 else 'negative' if sentiment[
                                                                                                       'compound'] < -0.05 else 'neutral'
                    })

    return mentions


# Apply the function to extract player mentions with context
posts_df['player_mentions'] = posts_df.apply(
    lambda row: extract_player_mentions_with_context(str(row['title']) + " " + str(row['selftext'])), axis=1)

# Create a flat dataframe of all player mentions
player_mentions_flat = []
for idx, row in posts_df.iterrows():
    post_id = idx
    for mention in row['player_mentions']:
        mention_data = {
            'post_id': post_id,
            'subreddit': row.get('subreddit', ''),
            'title': row['title'],
            'player': mention['player'],
            'context': mention['context'],
            'sentiment_compound': mention['sentiment_compound'],
            'sentiment': mention['sentiment'],
            'created_utc': row.get('created_utc', None),
            'score': row.get('score', None)
        }
        player_mentions_flat.append(mention_data)

player_df = pd.DataFrame(player_mentions_flat)

# Get overall sentiment stats
player_sentiment = player_df.groupby('player').agg({
    'sentiment_compound': ['mean', 'std', 'count'],
    'sentiment': lambda x: Counter(x)
}).reset_index()

# Flatten the MultiIndex columns
player_sentiment.columns = ['_'.join(col).strip('_') for col in player_sentiment.columns.values]

# Extract positive, negative, neutral counts
for sentiment_type in ['positive', 'negative', 'neutral']:
    player_sentiment[f'{sentiment_type}_count'] = player_sentiment['sentiment_<lambda>'].apply(
        lambda x: x.get(sentiment_type, 0))

# Calculate percentages
player_sentiment['total_mentions'] = player_sentiment['sentiment_compound_count']
for sentiment_type in ['positive', 'negative', 'neutral']:
    player_sentiment[f'{sentiment_type}_pct'] = (
            player_sentiment[f'{sentiment_type}_count'] / player_sentiment['total_mentions'] * 100
    )

# Calculate sentiment score (a weighted score based on positive vs negative ratio)
player_sentiment['sentiment_score'] = (
                                              (player_sentiment['positive_count'] * 1) +
                                              (player_sentiment['neutral_count'] * 0) +
                                              (player_sentiment['negative_count'] * -1)
                                      ) / player_sentiment['total_mentions']


# Extract most common words around players
def extract_keywords_for_player(player_name):
    # Get all context snippets for this player
    contexts = player_df[player_df['player'] == player_name]['context'].tolist()
    if not contexts:
        return {}

    # Combine all contexts
    text = ' '.join(contexts)

    # Use CountVectorizer to extract most common words (excluding the player name and common stopwords)
    stop_words = set(['the', 'a', 'an', 'and', 'is', 'in', 'to', 'of', 'for', 'on', 'with', 'at', 'from', 'by'])
    variations = set([v.lower() for v in real_madrid_players.get(player_name, [player_name])])
    stop_words.update(variations)

    vectorizer = CountVectorizer(stop_words=list(stop_words), ngram_range=(1, 2), max_features=15)
    X = vectorizer.fit_transform([text])

    # Get the most common words and their counts
    word_counts = dict(zip(vectorizer.get_feature_names_out(), X.toarray()[0]))
    return word_counts


# Add keyword analysis to each player
player_sentiment['top_keywords'] = player_sentiment['player'].apply(extract_keywords_for_player)


# Add a sentiment trend analysis function
def analyze_sentiment_trend(player_name):
    player_data = player_df[player_df['player'] == player_name]
    if 'created_utc' not in player_data.columns or len(player_data) < 5:
        return "Insufficient data for trend analysis"

    # Sort by timestamp
    player_data = player_data.sort_values('created_utc')

    # Calculate rolling sentiment
    rolling_sentiment = player_data['sentiment_compound'].rolling(window=5, min_periods=1).mean()

    # Determine trend direction
    first_third = rolling_sentiment.iloc[:len(rolling_sentiment) // 3].mean()
    last_third = rolling_sentiment.iloc[-len(rolling_sentiment) // 3:].mean()

    if last_third - first_third > 0.1:
        return "strongly improving"
    elif last_third - first_third > 0.05:
        return "improving"
    elif last_third - first_third < -0.1:
        return "strongly declining"
    elif last_third - first_third < -0.05:
        return "declining"
    else:
        return "stable"


# Add sentiment trend to each player
player_sentiment['sentiment_trend'] = player_sentiment['player'].apply(analyze_sentiment_trend)

# Sort by total mentions for the visualization
top_players_df = player_sentiment.sort_values('total_mentions', ascending=False).head(15)


# Create a comprehensive visualization function
def create_player_insights_visualizations(player_sentiment_df, output_prefix='player_analysis'):
    # Set the style
    sns.set(style="whitegrid")

    # 1. Create a horizontal stacked bar chart of sentiment percentages
    plt.figure(figsize=(12, 10))

    # Prepare data for stacked bar chart
    sentiment_data = player_sentiment_df.sort_values('total_mentions', ascending=False).head(15)

    # Create the stacked bar chart
    sentiment_data[['positive_pct', 'neutral_pct', 'negative_pct']].plot(
        kind='barh',
        stacked=True,
        color=['#2ecc71', '#95a5a6', '#e74c3c'],
        figsize=(12, 10)
    )

    plt.title('Sentiment Distribution for Most Mentioned Real Madrid Players', fontsize=16)
    plt.xlabel('Percentage', fontsize=14)
    plt.legend(['Positive', 'Neutral', 'Negative'], loc='upper right', fontsize=12)
    plt.yticks(range(len(sentiment_data)), sentiment_data['player'], fontsize=12)

    # Add total mentions as text
    for i, row in enumerate(sentiment_data.iterrows()):
        plt.text(
            101,
            i,
            f"Total: {int(row[1]['total_mentions'])}",
            va='center',
            fontsize=10
        )

    plt.tight_layout()
    plt.savefig(f'{output_prefix}_sentiment_distribution.png', dpi=300, bbox_inches='tight')

    # 2. Create individual player detailed insights - for top 5 players
    for _, player_row in sentiment_data.head(5).iterrows():
        player_name = player_row['player']

        # Create a figure with multiple subplots
        fig = plt.figure(figsize=(15, 10))

        # Add a title for the entire figure
        fig.suptitle(f'Player Insights: {player_name}', fontsize=20)

        # 1. Sentiment breakdown pie chart
        ax1 = plt.subplot2grid((2, 3), (0, 0))
        sentiment_counts = [
            player_row['positive_count'],
            player_row['neutral_count'],
            player_row['negative_count']
        ]
        ax1.pie(
            sentiment_counts,
            labels=['Positive', 'Neutral', 'Negative'],
            colors=['#2ecc71', '#95a5a6', '#e74c3c'],
            autopct='%1.1f%%',
            startangle=90
        )
        ax1.set_title('Sentiment Breakdown', fontsize=14)

        # 2. Word cloud of associated terms
        ax2 = plt.subplot2grid((2, 3), (0, 1), colspan=2)
        if player_row['top_keywords']:
            wordcloud = WordCloud(
                width=800,
                height=400,
                background_color='white',
                colormap='viridis',
                max_words=50
            ).generate_from_frequencies(player_row['top_keywords'])
            ax2.imshow(wordcloud, interpolation='bilinear')
            ax2.set_title('Most Associated Terms', fontsize=14)
            ax2.axis('off')
        else:
            ax2.text(0.5, 0.5, 'No keyword data available',
                     ha='center', va='center', fontsize=14)
            ax2.axis('off')

        # 3. Key stats text box
        ax3 = plt.subplot2grid((2, 3), (1, 0))
        stats_text = (
            f"Total Mentions: {int(player_row['total_mentions'])}\n"
            f"Average Sentiment: {player_row['sentiment_compound_mean']:.2f}\n"
            f"Sentiment Trend: {player_row['sentiment_trend']}\n"
            f"Positive/Negative Ratio: {player_row['positive_count'] / max(1, player_row['negative_count']):.2f}\n"
            f"Sentiment Volatility: {player_row['sentiment_compound_std']:.2f}"
        )
        ax3.text(0.5, 0.5, stats_text, ha='center', va='center', fontsize=12)
        ax3.set_title('Key Statistics', fontsize=14)
        ax3.axis('off')

        # 4. Most positive contexts
        ax4 = plt.subplot2grid((2, 3), (1, 1), colspan=1)
        positive_contexts = player_df[
            (player_df['player'] == player_name) &
            (player_df['sentiment'] == 'positive')
            ].sort_values('sentiment_compound', ascending=False)

        if len(positive_contexts) > 0:
            top_context = positive_contexts.iloc[0]['context']
            # Truncate if too long
            if len(top_context) > 300:
                top_context = top_context[:297] + '...'
            ax4.text(0.5, 0.5, top_context, ha='center', va='center',
                     wrap=True, fontsize=10)
            ax4.set_title('Most Positive Mention', fontsize=14)
        else:
            ax4.text(0.5, 0.5, 'No positive mentions found',
                     ha='center', va='center', fontsize=12)
            ax4.set_title('Most Positive Mention', fontsize=14)
        ax4.axis('off')

        # 5. Most negative contexts
        ax5 = plt.subplot2grid((2, 3), (1, 2), colspan=1)
        negative_contexts = player_df[
            (player_df['player'] == player_name) &
            (player_df['sentiment'] == 'negative')
            ].sort_values('sentiment_compound')

        if len(negative_contexts) > 0:
            top_context = negative_contexts.iloc[0]['context']
            # Truncate if too long
            if len(top_context) > 300:
                top_context = top_context[:297] + '...'
            ax5.text(0.5, 0.5, top_context, ha='center', va='center',
                     wrap=True, fontsize=10)
            ax5.set_title('Most Negative Mention', fontsize=14)
        else:
            ax5.text(0.5, 0.5, 'No negative mentions found',
                     ha='center', va='center', fontsize=12)
            ax5.set_title('Most Negative Mention', fontsize=14)
        ax5.axis('off')

        plt.tight_layout(rect=[0, 0, 1, 0.95])
        plt.savefig(f'{output_prefix}_{player_name.replace(" ", "_")}_insights.png',
                    dpi=300, bbox_inches='tight')
        plt.close()

    # 3. Create a comparison chart for top players
    plt.figure(figsize=(14, 8))

    # Sentiment score and mention count comparison
    top_comparisons = sentiment_data.head(10).copy()

    # Create a bubble chart
    plt.scatter(
        top_comparisons['sentiment_compound_mean'],
        top_comparisons['sentiment_score'],
        s=top_comparisons['total_mentions'] * 5,  # Size based on mention count
        alpha=0.7,
        c=top_comparisons['total_mentions'],
        cmap='viridis'
    )

    # Add player names as labels
    for _, row in top_comparisons.iterrows():
        plt.annotate(
            row['player'],
            xy=(row['sentiment_compound_mean'], row['sentiment_score']),
            xytext=(5, 5),
            textcoords='offset points',
            fontsize=11
        )

    plt.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
    plt.axvline(x=0, color='gray', linestyle='--', alpha=0.5)

    plt.title('Player Sentiment Analysis', fontsize=16)
    plt.xlabel('Average Sentiment (VADER compound score)', fontsize=14)
    plt.ylabel('Sentiment Score (positive-negative ratio)', fontsize=14)

    # Add a colorbar to indicate total mentions
    cbar = plt.colorbar()
    cbar.set_label('Total Mentions', fontsize=12)

    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f'{output_prefix}_player_comparison.png', dpi=300, bbox_inches='tight')

    print(f"Visualizations saved with prefix: {output_prefix}")


# Generate summary insights for each player
def generate_player_insights(player_data):
    insights = {}

    for _, row in player_data.sort_values('total_mentions', ascending=False).head(10).iterrows():
        player = row['player']

        # Format the top keywords as a readable string
        top_keywords_str = ", ".join([
            f"{word} ({count})" for word, count in
            sorted(row['top_keywords'].items(), key=lambda x: x[1], reverse=True)[:5]
        ]) if row['top_keywords'] else "No significant keywords found"

        # Create player insight
        insight = {
            "total_mentions": int(row['total_mentions']),
            "sentiment_score": round(row['sentiment_compound_mean'], 2),
            "sentiment_distribution": {
                "positive": round(row['positive_pct'], 1),
                "neutral": round(row['neutral_pct'], 1),
                "negative": round(row['negative_pct'], 1)
            },
            "top_keywords": top_keywords_str,
            "sentiment_trend": row['sentiment_trend'],
            "volatility": round(row['sentiment_compound_std'], 2),
            "summary": f"{player} was mentioned {int(row['total_mentions'])} times with an overall "
                       f"sentiment of {round(row['sentiment_compound_mean'], 2)} "
                       f"({row['sentiment_trend']} trend). "
                       f"Comments were {round(row['positive_pct'], 1)}% positive, "
                       f"{round(row['neutral_pct'], 1)}% neutral, and "
                       f"{round(row['negative_pct'], 1)}% negative. "
                       f"Top associated terms: {top_keywords_str}."
        }

        insights[player] = insight

    return insights


# Main analysis function
def analyze_player_data():
    # Create visualizations
    create_player_insights_visualizations(player_sentiment)

    # Generate player insights
    insights = generate_player_insights(player_sentiment)

    # Return the complete analysis results
    return {
        "player_sentiment_df": player_sentiment,
        "player_mentions_df": player_df,
        "player_insights": insights
    }


# Execute the analysis
analysis_results = analyze_player_data()

# Print a summary of the top 5 players
print("===== PLAYER ANALYSIS SUMMARY =====")
for player, insight in list(analysis_results["player_insights"].items())[:5]:
    print(f"\n{player.upper()}:")
    print(insight["summary"])
    print(f"Sentiment trend: {insight['sentiment_trend']}")
    print("-" * 50)

# Save the player sentiment data
player_sentiment.to_csv('player_sentiment_analysis.csv', index=False)

# Return the DataFrames and insights for further analysis
player_sentiment_df = analysis_results["player_sentiment_df"]
player_mentions_df = analysis_results["player_mentions_df"]
player_insights = analysis_results["player_insights"]


'''
===== PLAYER ANALYSIS SUMMARY =====

MBAPPÉ:
Mbappé was mentioned 28 times with an overall sentiment of -0.05 (stable trend). Comments were 10.7% positive, 71.4% neutral, and 17.9% negative. Top associated terms: madrid (22), real (22), real madrid (22), city (9), city real (9).
Sentiment trend: stable
--------------------------------------------------

VALVERDE:
Valverde was mentioned 25 times with an overall sentiment of -0.02 (strongly improving trend). Comments were 12.0% positive, 68.0% neutral, and 20.0% negative. Top associated terms: p82 (19), sprite6 (19), sprite6 p82 (19), ceballos (8), icon (8).
Sentiment trend: strongly improving
--------------------------------------------------

MODRIC:
Modric was mentioned 20 times with an overall sentiment of 0.06 (stable trend). Comments were 10.0% positive, 90.0% neutral, and 0.0% negative. Top associated terms: sprite6 (12), icon (8), substitution (8), ivanušec (6), 00 (5).
Sentiment trend: stable
--------------------------------------------------

ANCELOTTI:
Ancelotti was mentioned 20 times with an overall sentiment of -0.04 (strongly improving trend). Comments were 35.0% positive, 30.0% neutral, and 35.0% negative. Top associated terms: losses (15), sprite6 (13), p82 (10), sprite6 p82 (10), guardiola (9).
Sentiment trend: strongly improving
--------------------------------------------------

BELLINGHAM:
Bellingham was mentioned 19 times with an overall sentiment of 0.12 (improving trend). Comments were 36.8% positive, 52.6% neutral, and 10.5% negative. Top associated terms: it (8), madrid (8), real (8), real madrid (8), but (5).
Sentiment trend: improving
--------------------------------------------------
'''