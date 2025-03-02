import pandas as pd

# Load the comments CSV
comments_df = pd.read_csv("champions_league_comments.csv")

# Display the first few rows
print(comments_df.head())

################### Extract entities
import spacy

nlp = spacy.load("en_core_web_sm")

def extract_entities(text):
    """Extract named entities related to teams or players."""
    doc = nlp(text)
    entities = [ent.text for ent in doc.ents if ent.label_ in ["PERSON", "ORG"]]
    return entities

# Apply entity extraction
comments_df["entities_extracted"] = comments_df["body"].astype(str).apply(extract_entities)

# Display results
print(comments_df[["body", "entities_extracted"]].head())

################### Sentiments
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

analyzer = SentimentIntensityAnalyzer()

def analyze_sentiment(text):
    """Analyze sentiment using VADER."""
    score = analyzer.polarity_scores(text)["compound"]
    return "positive" if score > 0.05 else "negative" if score < -0.05 else "neutral"

# Apply sentiment analysis
comments_df["sentiment"] = comments_df["body"].astype(str).apply(analyze_sentiment)

# Display results
print(comments_df[["body", "sentiment"]].head())


structured_data = comments_df

# Save to a new CSV
structured_data.to_csv("structured_champions_league_comments.csv", index=False)

print(structured_data)
print("Structured data saved successfully!")
