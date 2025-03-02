import pandas as pd
import re
import spacy
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict, Counter
import numpy as np
from wordcloud import WordCloud
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Load spaCy model for entity recognition
try:
    nlp = spacy.load("en_core_web_sm")
except:
    print("You need to install the spaCy model first. Run: python -m spacy download en_core_web_sm")
    exit(1)

# List of top Champions League teams to look for
CL_TEAMS = [
    "Real Madrid", "Barcelona", "Bayern Munich", "Liverpool", "Manchester City",
    "PSG", "Chelsea", "Juventus", "Borussia Dortmund", "Atletico Madrid",
    "Inter Milan", "AC Milan", "Arsenal", "Ajax", "Benfica", "Porto",
    "Manchester United", "Napoli", "Roma", "Bayer Leverkusen", "RB Leipzig",
    "Tottenham", "Monaco", "Lyon", "Marseille", "Sporting CP",
    "Celtic", "Rangers", "Galatasaray", "Fenerbahce"
]

# Performance metrics to extract
PERFORMANCE_METRICS = [
    r'(\d+)[\s-]*(\d+)',  # Score pattern (e.g., "2-1", "3 0")
    r'(\d+)(?:\s*-\s*\d+)?\s*win',  # Win with score (e.g., "2-1 win", "3-0 win")
    r'(\d+)(?:\s*-\s*\d+)?\s*loss',  # Loss with score
    r'(\d+)(?:\s*-\s*\d+)?\s*defeat',  # Defeat with score
    r'(\d+)\s*goals?\s*scored',  # Goals scored
    r'(\d+)\s*goals?\s*conceded',  # Goals conceded
    r'(\d+)\s*shots',  # Shots
    r'(\d+)%\s*possession',  # Possession percentage
    r'(\d+)\s*xG',  # Expected goals
    r'(\d+)\s*saves',  # Saves
    r'(\d+)\s*yellow cards?',  # Yellow cards
    r'(\d+)\s*red cards?',  # Red cards
    r'(\d+)\s*fouls?',  # Fouls
    r'(\d+)\s*corner',  # Corners
    r'(\d+)\s*chances?',  # Chances created
]


def load_comments_data(file_path):
    """Load comments data from CSV file."""
    try:
        return pd.read_csv(file_path)
    except FileNotFoundError:
        print(f"Error: File {file_path} not found.")
        return None


def preprocess_team_names():
    """Create variations of team names for better matching."""
    team_variations = {}
    for team in CL_TEAMS:
        variations = [team.lower()]

        # Add common variations
        if "Manchester" in team:
            if "City" in team:
                variations.extend(["man city", "city", "mcfc", "manchester city"])
            else:
                variations.extend(["man united", "man utd", "united", "mufc", "manchester united"])
        elif "Real Madrid" in team:
            variations.extend(["real", "madrid", "los blancos", "rmcf"])
        elif "Barcelona" in team:
            variations.extend(["barca", "fcb", "blaugrana"])
        elif "Bayern" in team:
            variations.extend(["bayern", "fcb", "fcbayern"])
        elif "Liverpool" in team:
            variations.extend(["lfc", "the reds", "liverpool fc"])
        elif "Juventus" in team:
            variations.extend(["juve", "la vecchia signora", "old lady"])
        elif "Borussia Dortmund" in team:
            variations.extend(["bvb", "dortmund", "bvb 09"])
        elif "Atletico Madrid" in team:
            variations.extend(["atleti", "atletico", "colchoneros"])
        elif "Inter Milan" in team:
            variations.extend(["inter", "nerazzurri"])
        elif "AC Milan" in team:
            variations.extend(["milan", "acm", "rossoneri"])

        team_variations[team] = list(set(variations))

    return team_variations


def extract_team_mentions(comments_df, team_variations):
    """Extract team mentions from comments."""
    team_mentions = defaultdict(int)
    team_contexts = defaultdict(list)

    # Combine all comments into a single string for processing
    for _, row in comments_df.iterrows():
        if pd.isna(row['body']):
            continue

        comment = row['body'].lower()

        # Check for team mentions
        for team, variations in team_variations.items():
            for variation in variations:
                if re.search(r'\b' + re.escape(variation) + r'\b', comment):
                    team_mentions[team] += 1

                    # Extract 150 characters around the team mention for context
                    match_positions = [m.start() for m in re.finditer(r'\b' + re.escape(variation) + r'\b', comment)]
                    for pos in match_positions:
                        start = max(0, pos - 75)
                        end = min(len(comment), pos + 75)
                        context = comment[start:end]
                        team_contexts[team].append(context)

                    break  # Count each team only once per comment

    return team_mentions, team_contexts


def extract_performance_metrics(team_contexts):
    """Extract performance metrics for each team."""
    team_metrics = defaultdict(lambda: defaultdict(list))

    for team, contexts in team_contexts.items():
        for context in contexts:
            # Check for score patterns
            score_matches = re.findall(r'(\d+)\s*-\s*(\d+)', context)
            if score_matches:
                for goals_for, goals_against in score_matches:
                    team_metrics[team]['scores'].append((int(goals_for), int(goals_against)))

            # Check for other metrics
            for metric_name, metric_pattern in [
                ('wins', r'\b' + re.escape(team.lower()) + r'.*?\bwon\b'),
                ('losses', r'\b' + re.escape(team.lower()) + r'.*?\blost\b'),
                ('goals_scored', r'\b' + re.escape(team.lower()) + r'.*?(\d+)\s*goals?\s*'),
                ('possession', r'\b' + re.escape(team.lower()) + r'.*?(\d+)%\s*possession'),
                ('shots', r'\b' + re.escape(team.lower()) + r'.*?(\d+)\s*shots?'),
                ('corners', r'\b' + re.escape(team.lower()) + r'.*?(\d+)\s*corners?'),
                ('clean_sheets', r'\b' + re.escape(team.lower()) + r'.*?clean\s*sheets?'),
            ]:
                matches = re.search(metric_pattern, context)
                if matches:
                    if metric_name in ['goals_scored', 'possession', 'shots', 'corners'] and matches.groups():
                        try:
                            team_metrics[team][metric_name].append(int(matches.group(1)))
                        except (ValueError, IndexError):
                            pass
                    else:
                        team_metrics[team][metric_name].append(1)

    # Process the collected metrics
    processed_metrics = {}
    for team, metrics in team_metrics.items():
        processed_metrics[team] = {}

        # Process scores to get win/loss/draw counts and average goals
        if 'scores' in metrics:
            wins = sum(1 for gf, ga in metrics['scores'] if gf > ga)
            losses = sum(1 for gf, ga in metrics['scores'] if gf < ga)
            draws = sum(1 for gf, ga in metrics['scores'] if gf == ga)

            processed_metrics[team]['wins'] = wins + len(metrics.get('wins', []))
            processed_metrics[team]['losses'] = losses + len(metrics.get('losses', []))
            processed_metrics[team]['draws'] = draws

            goals_for = sum(gf for gf, _ in metrics['scores'])
            goals_against = sum(ga for _, ga in metrics['scores'])

            processed_metrics[team]['goals_for'] = goals_for + sum(metrics.get('goals_scored', []))
            processed_metrics[team]['goals_against'] = goals_against
        else:
            processed_metrics[team]['wins'] = len(metrics.get('wins', []))
            processed_metrics[team]['losses'] = len(metrics.get('losses', []))
            processed_metrics[team]['draws'] = 0
            processed_metrics[team]['goals_for'] = sum(metrics.get('goals_scored', []))
            processed_metrics[team]['goals_against'] = 0

        # Process other metrics
        for metric in ['possession', 'shots', 'corners']:
            if metric in metrics and metrics[metric]:
                processed_metrics[team][metric] = sum(metrics[metric]) / len(metrics[metric])
            else:
                processed_metrics[team][metric] = 0

        processed_metrics[team]['clean_sheets'] = len(metrics.get('clean_sheets', []))

    return processed_metrics


def extract_sentiment_by_team(comments_df, team_variations):
    """Analyze sentiment for each team mention."""
    from nltk.sentiment import SentimentIntensityAnalyzer
    import nltk

    try:
        nltk.data.find('sentiment/vader_lexicon.zip')
    except LookupError:
        nltk.download('vader_lexicon')

    sia = SentimentIntensityAnalyzer()
    team_sentiments = defaultdict(list)

    for _, row in comments_df.iterrows():
        if pd.isna(row['body']):
            continue

        comment = row['body']

        # First check which teams are mentioned
        mentioned_teams = set()
        for team, variations in team_variations.items():
            for variation in variations:
                if re.search(r'\b' + re.escape(variation) + r'\b', comment.lower()):
                    mentioned_teams.add(team)

        if not mentioned_teams:
            continue

        # Split comment into sentences for more granular analysis
        sentences = re.split(r'(?<=[.!?])\s+', comment)

        for sentence in sentences:
            sentiment = sia.polarity_scores(sentence)

            # Check which teams are mentioned in this sentence
            sentence_teams = set()
            for team in mentioned_teams:
                for variation in team_variations[team]:
                    if re.search(r'\b' + re.escape(variation) + r'\b', sentence.lower()):
                        sentence_teams.add(team)

            # Assign sentiment to mentioned teams
            for team in sentence_teams:
                team_sentiments[team].append(sentiment['compound'])

    # Calculate average sentiment for each team
    avg_sentiments = {}
    for team, sentiments in team_sentiments.items():
        if sentiments:
            avg_sentiments[team] = sum(sentiments) / len(sentiments)
        else:
            avg_sentiments[team] = 0

    return avg_sentiments, team_sentiments


def analyze_performance_trends(comments_df, team_variations):
    """Analyze performance trends over time for teams."""
    team_performance_over_time = defaultdict(lambda: defaultdict(list))

    # Convert timestamp to datetime
    comments_df['created_utc'] = pd.to_datetime(comments_df['created_utc'])

    # Group by date
    comments_df['date'] = comments_df['created_utc'].dt.date

    # Group comments by date
    for date, group in comments_df.groupby('date'):
        # Combine all comments for this date
        date_comments = ' '.join(group['body'].fillna(''))

        # Check for team mentions
        for team, variations in team_variations.items():
            mention_count = 0
            for variation in variations:
                mention_count += len(re.findall(r'\b' + re.escape(variation) + r'\b', date_comments.lower()))

            team_performance_over_time[team]['mentions'].append((date, mention_count))

            # Check for performance indicators
            wins = len(re.findall(r'\b' + '|'.join(re.escape(v) for v in variations) + r'\b.*?\bwon\b|\bwin\b',
                                  date_comments.lower()))
            losses = len(re.findall(r'\b' + '|'.join(re.escape(v) for v in variations) + r'\b.*?\blost\b|\bloss\b',
                                    date_comments.lower()))

            team_performance_over_time[team]['wins'].append((date, wins))
            team_performance_over_time[team]['losses'].append((date, losses))

    return team_performance_over_time


def create_visualizations(team_mentions, processed_metrics, avg_sentiments, team_performance_over_time):
    """Create visualizations for the extracted data."""
    # 1. Most mentioned teams
    top_teams = sorted(team_mentions.items(), key=lambda x: x[1], reverse=True)[:10]
    top_teams_df = pd.DataFrame(top_teams, columns=['Team', 'Mentions'])

    # Create bar chart for team mentions
    plt.figure(figsize=(12, 6))
    sns.barplot(x='Mentions', y='Team', data=top_teams_df)
    plt.title('Most Mentioned Champions League Teams')
    plt.tight_layout()
    plt.savefig('team_mentions.png')

    # 2. Performance metrics
    if processed_metrics:
        # Filter for teams with actual performance data
        teams_with_data = {team: metrics for team, metrics in processed_metrics.items()
                           if metrics.get('wins', 0) > 0 or metrics.get('losses', 0) > 0 or metrics.get('goals_for',
                                                                                                        0) > 0}

        if teams_with_data:
            # Create dataframe for performance metrics
            performance_data = []
            for team, metrics in teams_with_data.items():
                performance_data.append({
                    'Team': team,
                    'Wins': metrics.get('wins', 0),
                    'Draws': metrics.get('draws', 0),
                    'Losses': metrics.get('losses', 0),
                    'Goals For': metrics.get('goals_for', 0),
                    'Goals Against': metrics.get('goals_against', 0),
                    'Possession': metrics.get('possession', 0),
                    'Mentions': team_mentions.get(team, 0)
                })

            perf_df = pd.DataFrame(performance_data)

            # Sort by total mentions for consistency
            perf_df = perf_df.sort_values('Mentions', ascending=False).head(10)

            # Create a stacked bar chart for wins/draws/losses
            plt.figure(figsize=(12, 6))
            perf_df_melted = pd.melt(perf_df, id_vars=['Team'], value_vars=['Wins', 'Draws', 'Losses'])
            sns.barplot(x='Team', y='value', hue='variable', data=perf_df_melted)
            plt.title('Performance of Champions League Teams (Based on Reddit Comments)')
            plt.xticks(rotation=45, ha='right')
            plt.legend(title='Result')
            plt.tight_layout()
            plt.savefig('team_performance.png')

            # Create a goals chart
            plt.figure(figsize=(12, 6))
            goals_df = perf_df[['Team', 'Goals For', 'Goals Against']].sort_values('Goals For', ascending=False)
            goals_df_melted = pd.melt(goals_df, id_vars=['Team'], value_vars=['Goals For', 'Goals Against'])
            sns.barplot(x='Team', y='value', hue='variable', data=goals_df_melted)
            plt.title('Goals For and Against Champions League Teams (Based on Reddit Comments)')
            plt.xticks(rotation=45, ha='right')
            plt.legend(title='Goals')
            plt.tight_layout()
            plt.savefig('team_goals.png')

    # 3. Sentiment analysis by team
    sentiment_df = pd.DataFrame([{"Team": team, "Sentiment": sentiment}
                                 for team, sentiment in avg_sentiments.items()
                                 if team in [t[0] for t in top_teams]])

    plt.figure(figsize=(12, 6))
    bars = sns.barplot(x='Team', y='Sentiment', data=sentiment_df,
                       palette=sns.color_palette("RdYlGn", n_colors=len(sentiment_df)))
    plt.title('Fan Sentiment Towards Champions League Teams')
    plt.axhline(y=0, color='gray', linestyle='-', alpha=0.3)
    plt.xticks(rotation=45, ha='right')

    # Add value labels on the bars
    for i, bar in enumerate(bars.patches):
        bars.text(bar.get_x() + bar.get_width() / 2.,
                  bar.get_height() + 0.03 if bar.get_height() >= 0 else bar.get_height() - 0.08,
                  f'{bar.get_height():.2f}',
                  ha='center', va='bottom' if bar.get_height() >= 0 else 'top')

    plt.tight_layout()
    plt.savefig('team_sentiment.png')

    # 4. Create an interactive Plotly visualization
    if processed_metrics:
        # Create a dataframe for the bubble chart
        bubble_data = []
        for team, metrics in processed_metrics.items():
            if team in team_mentions and team_mentions[team] > 5:  # Filter for relevance
                bubble_data.append({
                    'Team': team,
                    'Mentions': team_mentions[team],
                    'Wins': metrics.get('wins', 0),
                    'Losses': metrics.get('losses', 0),
                    'Goals': metrics.get('goals_for', 0),
                    'Sentiment': avg_sentiments.get(team, 0)
                })

        if bubble_data:
            bubble_df = pd.DataFrame(bubble_data)

            # Create interactive bubble chart
            fig = px.scatter(
                bubble_df,
                x='Wins',
                y='Sentiment',
                size='Mentions',
                color='Goals',
                hover_name='Team',
                text='Team',
                title='Champions League Teams: Performance vs Sentiment',
                size_max=60,
                color_continuous_scale='Viridis'
            )

            fig.update_traces(textposition='top center')
            fig.update_layout(
                xaxis_title='Perceived Wins (from comments)',
                yaxis_title='Sentiment Score (-1 to 1)',
                coloraxis_colorbar_title='Perceived Goals'
            )

            # Save the interactive chart as HTML
            fig.write_html("team_performance_interactive.html")

    # 5. Team performance over time (for top mentioned teams)
    # Create multi-line chart for team mentions over time
    top_5_teams = [team for team, _ in top_teams[:5]]

    # Create a time series plot for the top 5 teams
    plt.figure(figsize=(14, 7))

    for team in top_5_teams:
        if team in team_performance_over_time:
            mentions_over_time = team_performance_over_time[team]['mentions']
            if mentions_over_time:
                dates, counts = zip(*sorted(mentions_over_time))
                plt.plot(dates, counts, marker='o', linewidth=2, label=team)

    plt.title('Mentions of Top Champions League Teams Over Time')
    plt.xlabel('Date')
    plt.ylabel('Number of Mentions')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig('team_mentions_over_time.png')

    # Create word cloud for most discussed topics
    if team_mentions:
        wordcloud = WordCloud(width=800, height=400, background_color='white',
                              colormap='viridis', max_words=100).generate_from_frequencies(team_mentions)

        plt.figure(figsize=(10, 5))
        plt.imshow(wordcloud, interpolation='bilinear')
        plt.axis('off')
        plt.tight_layout()
        plt.savefig('team_wordcloud.png')

    # Return a summary of findings
    summary = {
        'top_teams': [team for team, _ in top_teams[:5]],
        'top_performing_teams': sorted(
            [(team, metrics.get('wins', 0), metrics.get('goals_for', 0))
             for team, metrics in processed_metrics.items() if metrics.get('wins', 0) > 0],
            key=lambda x: (x[1], x[2]), reverse=True
        )[:5] if processed_metrics else [],
        'most_positive_teams': sorted(avg_sentiments.items(), key=lambda x: x[1], reverse=True)[:3],
        'most_negative_teams': sorted(avg_sentiments.items(), key=lambda x: x[1])[:3]
    }

    return summary


def main():
    # Load comments data
    comments_df = load_comments_data('champions_league_comments.csv')
    if comments_df is None:
        print("Could not load comments data. Exiting.")
        return

    print(f"Loaded {len(comments_df)} comments for analysis")

    # Preprocess team names for better matching
    team_variations = preprocess_team_names()

    # Extract team mentions
    print("Extracting team mentions...")
    team_mentions, team_contexts = extract_team_mentions(comments_df, team_variations)
    print(f"Found {sum(team_mentions.values())} mentions of {len(team_mentions)} different teams")

    # Extract performance metrics
    print("Extracting performance metrics...")
    processed_metrics = extract_performance_metrics(team_contexts)
    print(f"Extracted performance metrics for {len(processed_metrics)} teams")

    # Extract sentiment by team
    print("Analyzing sentiment by team...")
    avg_sentiments, team_sentiments = extract_sentiment_by_team(comments_df, team_variations)
    print(f"Analyzed sentiment for {len(avg_sentiments)} teams")

    # Analyze performance trends over time
    print("Analyzing performance trends over time...")
    team_performance_over_time = analyze_performance_trends(comments_df, team_variations)

    # Create visualizations
    print("Creating visualizations...")
    summary = create_visualizations(team_mentions, processed_metrics, avg_sentiments, team_performance_over_time)

    # Create a performance metrics dataframe
    performance_df = pd.DataFrame([
        {
            'Team': team,
            'Mentions': team_mentions.get(team, 0),
            'Wins': metrics.get('wins', 0),
            'Draws': metrics.get('draws', 0),
            'Losses': metrics.get('losses', 0),
            'Goals For': metrics.get('goals_for', 0),
            'Goals Against': metrics.get('goals_against', 0),
            'Sentiment': avg_sentiments.get(team, 0)
        }
        for team, metrics in processed_metrics.items()
    ])

    # Save performance metrics to CSV
    performance_df.to_csv('team_performance_metrics.csv', index=False)
    print("Saved team performance metrics to team_performance_metrics.csv")

    # Print summary
    print("\nSummary of Findings:")
    print("Top mentioned teams:", ", ".join(summary['top_teams']))

    if summary['top_performing_teams']:
        print("\nTop performing teams (based on perceived wins and goals):")
        for team, wins, goals in summary['top_performing_teams']:
            print(f"  - {team}: {wins} wins, {goals} goals")

    print("\nTeams with most positive sentiment:")
    for team, sentiment in summary['most_positive_teams']:
        print(f"  - {team}: {sentiment:.2f}")

    print("\nTeams with most negative sentiment:")
    for team, sentiment in summary['most_negative_teams']:
        print(f"  - {team}: {sentiment:.2f}")

    print("\nVisualizations saved to:")
    print("- team_mentions.png")
    print("- team_performance.png")
    print("- team_goals.png")
    print("- team_sentiment.png")
    print("- team_mentions_over_time.png")
    print("- team_wordcloud.png")
    print("- team_performance_interactive.html")


if __name__ == "__main__":
    main()