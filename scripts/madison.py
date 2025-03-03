import os
from dotenv import load_dotenv
import praw
from wordcloud import WordCloud
import matplotlib.pyplot as plt
from collections import Counter
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
from textblob import TextBlob
import pandas as pd
from datetime import datetime
import re
import string

# Ensure NLTK resources are downloaded properly
nltk.download('punkt')  # Not punkt_tab, just punkt
nltk.download('stopwords')
nltk.download('wordnet')


# Function to get sentiment polarity
def get_sentiment(text):
    blob = TextBlob(text)
    return blob.sentiment.polarity  # Range from -1 (negative) to 1 (positive)


# Load environment variables
load_dotenv()

# Configure Reddit API
reddit = praw.Reddit(
    client_id=os.getenv('REDDIT_CLIENT_ID'),
    client_secret=os.getenv('REDDIT_CLIENT_SECRET'),
    user_agent=os.getenv('USER_AGENT')
)

# Search query and subreddit - broadened to catch more AI-related content
query = "UW-Madison AI"  # Broadened query
subreddit = reddit.subreddit("all")
search_limit = 100  # Increased limit to get more data

# Create output directory if it doesn't exist
output_dir = 'uw_madison_ai_topics'
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# Store submissions to avoid multiple API calls
print("Fetching submissions from Reddit...")
submissions_list = list(subreddit.search(query, sort="new", limit=search_limit))

# Print debug information
print(f"Number of submissions found: {len(submissions_list)}")

# Extract and preprocess text content
full_texts = []
text_content = ""
ai_keywords = ["ai", "artificial intelligence", "machine learning", "deep learning", "neural network",
               "nlp", "computer vision", "robotics", "data science", "algorithm",
               "chatgpt", "claude", "gpt", "llm", "large language model", "transformer",
               "generative ai", "ml", "automation", "big data", "predictive analytics"]

# Additional UW-Madison specific keywords
uw_ai_keywords = ["uw ai", "wisconsin ai", "uw machine learning",
                  "uw research", "madison tech", "uw computer science", "uw-madison ai",
                  "uw-madison machine learning", "uw-madison research", "wisconsin ai",
                  "wisconsin machine learning",
                  "wisconsin computer science", "wisconsin-madison ai",
                  "wisconsin-madison machine learning", "wisconsin-madison research",
                  "wisconsin ai", "wisconsin machine learning",
                  "wisconsin tech", "wisconsin computer science", "wisconsin-madison ai",
                  "wisconsin-madison machine learning",
                  "wisconsin-madison computer science", "open-source"]

# Prepare for topic modeling
documents = []
post_data = []
ai_related_posts = []


# Simple tokenization function without relying on nltk's word_tokenize
def simple_tokenize(text):
    # Remove punctuation
    text = text.translate(str.maketrans('', '', string.punctuation))
    # Convert to lowercase and split on whitespace
    return text.lower().split()


# Custom preprocessing that doesn't rely on nltk's word_tokenize
def custom_preprocess(text):
    # Convert to lowercase
    text = text.lower()
    # Remove punctuation
    text = re.sub(r'[^\w\s]', '', text)
    # Tokenize by splitting on whitespace
    tokens = text.split()
    # Remove stopwords
    stop_words = set(stopwords.words('english')) if nltk.data.find('corpora/stopwords') else set()
    custom_stops = ['uw', 'madison', 'wisconsin', 'university']
    all_stops = stop_words.union(custom_stops)

    # Filter tokens
    filtered_tokens = [token for token in tokens if token not in all_stops and len(token) > 2]

    # Only lemmatize if we can
    try:
        lemmatizer = WordNetLemmatizer()
        filtered_tokens = [lemmatizer.lemmatize(token) for token in filtered_tokens]
    except LookupError:
        # If lemmatization fails, just use the filtered tokens
        pass

    return ' '.join(filtered_tokens)


for submission in submissions_list:
    # Combine title and text
    combined_text = submission.title + " " + submission.selftext
    text_content += combined_text + " "

    # Check if the post is AI-related
    is_ai_related = False
    lower_text = combined_text.lower()

    # Check for AI keywords
    for keyword in ai_keywords + uw_ai_keywords:
        if keyword in lower_text:
            is_ai_related = True
            break

    # If AI-related, add to our collection
    if is_ai_related:
        ai_related_posts.append(submission)
        documents.append(combined_text)

        # Add sentiment and metadata
        sentiment = get_sentiment(combined_text)
        post_data.append({
            'title': submission.title,
            'comments': submission.num_comments,
            'upvotes': submission.score,
            'sentiment': sentiment,
            'date': datetime.fromtimestamp(submission.created_utc),
            'url': f"https://www.reddit.com{submission.permalink}"
        })

print(f"Number of AI-related posts found: {len(ai_related_posts)}")

# Only proceed if we have AI-related posts
if ai_related_posts:
    try:
        # Preprocess documents with our safer function
        print("Preprocessing documents...")
        processed_docs = [custom_preprocess(doc) for doc in documents]

        # Vectorize
        print("Vectorizing documents...")
        vectorizer = CountVectorizer(max_df=0.95, min_df=2, max_features=1000)
        X = vectorizer.fit_transform(processed_docs)

        # Apply LDA
        num_topics = min(5, len(processed_docs))  # Set reasonable number of topics
        print(f"Extracting {num_topics} topics using LDA...")
        lda = LatentDirichletAllocation(n_components=num_topics, random_state=42)
        lda.fit(X)

        # Get feature names
        feature_names = vectorizer.get_feature_names_out()

        # Print topics
        print("\n===== AI TOPICS AT UW-MADISON MENTIONED ON REDDIT =====")
        for topic_idx, topic in enumerate(lda.components_):
            top_words_idx = topic.argsort()[:-11:-1]  # Get top 10 words
            top_words = [feature_names[i] for i in top_words_idx]
            print(f"Topic #{topic_idx + 1}: {', '.join(top_words)}")

        # Save topics to file
        with open(os.path.join(output_dir, 'uw_madison_ai_topics.txt'), 'w') as f:
            f.write("UW-MADISON AI TOPICS MENTIONED ON REDDIT\n")
            f.write("=" * 50 + "\n\n")
            for topic_idx, topic in enumerate(lda.components_):
                top_words_idx = topic.argsort()[:-11:-1]
                top_words = [feature_names[i] for i in top_words_idx]
                f.write(f"Topic #{topic_idx + 1}: {', '.join(top_words)}\n\n")

        # Create word cloud of AI terms
        if text_content.strip():
            print("Generating word cloud...")
            # Simple tokenization and filtering for the word cloud
            tokens = simple_tokenize(text_content)
            stop_words = set(stopwords.words("english")) if nltk.data.find('corpora/stopwords') else set()
            filtered_words = [word for word in tokens if word not in stop_words and len(word) > 2]

            filtered_text = " ".join(filtered_words)

            # Generate the Word Cloud
            wordcloud = WordCloud(width=800, height=400, background_color='white',
                                  colormap='plasma', min_word_length=3).generate(filtered_text)

            # Display the Word Cloud
            plt.figure(figsize=(10, 5))
            plt.imshow(wordcloud, interpolation='bilinear')
            plt.axis("off")
            plt.title("UW-Madison AI Topics on Reddit", fontsize=14)
            plt.savefig(os.path.join(output_dir, 'uw_madison_ai_wordcloud.png'))
            plt.close()

        # Create a DataFrame from post data
        df = pd.DataFrame(post_data)

        # Save the detailed post information
        df.to_csv(os.path.join(output_dir, 'uw_madison_ai_posts.csv'), index=False)

        # Additional analysis: Top AI posts by engagement
        if not df.empty:
            print("\n===== TOP UW-MADISON AI POSTS BY ENGAGEMENT =====")
            top_posts = df.sort_values(by='comments', ascending=False).head(5)
            for i, (_, post) in enumerate(top_posts.iterrows()):
                print(f"{i + 1}. {post['title']} - {post['comments']} comments, {post['upvotes']} upvotes")
                print(f"   URL: {post['url']}")

        # Create visualization of topic distribution
        topic_distribution = lda.transform(X)
        topic_names = [f"Topic {i + 1}" for i in range(num_topics)]

        # Calculate average topic presence
        topic_presence = topic_distribution.mean(axis=0)

        # Plot topic distribution
        plt.figure(figsize=(10, 6))
        plt.bar(topic_names, topic_presence, color='skyblue')
        plt.title('AI Topic Distribution in UW-Madison Reddit Posts')
        plt.xlabel('Topics')
        plt.ylabel('Average Topic Presence')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'topic_distribution.png'))
        plt.close()

        # Save individual topics and their representative posts
        for topic_idx in range(num_topics):
            # Get top documents for this topic
            doc_topic_probs = topic_distribution[:, topic_idx]
            top_doc_indices = doc_topic_probs.argsort()[::-1][:3]  # Top 3 docs

            with open(os.path.join(output_dir, f'topic_{topic_idx + 1}_details.txt'), 'w') as f:
                top_words_idx = lda.components_[topic_idx].argsort()[:-11:-1]
                top_words = [feature_names[i] for i in top_words_idx]

                f.write(f"TOPIC #{topic_idx + 1}: {', '.join(top_words)}\n")
                f.write("=" * 50 + "\n\n")
                f.write("Representative Posts:\n\n")

                for doc_idx in top_doc_indices:
                    if doc_idx < len(ai_related_posts):
                        post = ai_related_posts[doc_idx]
                        f.write(f"Title: {post.title}\n")
                        f.write(f"Score: {post.score}\n")
                        f.write(f"Comments: {post.num_comments}\n")
                        f.write(f"URL: https://www.reddit.com{post.permalink}\n")
                        f.write(f"Text excerpt: {post.selftext[:500]}...\n\n")

        print(f"\nAnalysis complete! Results saved to the '{output_dir}' directory.")

    except Exception as e:
        print(f"Error during topic extraction: {str(e)}")
        # Fallback analysis if topic modeling fails
        print("Performing fallback keyword-based analysis...")

        # Simple keyword counting
        ai_keyword_counts = Counter()
        for post in ai_related_posts:
            combined_text = (post.title + " " + post.selftext).lower()
            for keyword in ai_keywords:
                if keyword in combined_text:
                    ai_keyword_counts[keyword] += 1

        # Print and save most common AI keywords
        print("\n===== MOST COMMON AI KEYWORDS IN UW-MADISON POSTS =====")
        for keyword, count in ai_keyword_counts.most_common(10):
            print(f"{keyword}: {count} mentions")

        with open(os.path.join(output_dir, 'ai_keyword_counts.txt'), 'w') as f:
            f.write("AI KEYWORDS MENTIONED IN UW-MADISON REDDIT POSTS\n")
            f.write("=" * 50 + "\n\n")
            for keyword, count in ai_keyword_counts.most_common():
                f.write(f"{keyword}: {count} mentions\n")

        # Save post data
        if post_data:
            df = pd.DataFrame(post_data)
            df.to_csv(os.path.join(output_dir, 'uw_madison_ai_posts.csv'), index=False)

else:
    print("No AI-related posts found for UW-Madison.")
    print("Try broadening your search query or check your Reddit API credentials.")