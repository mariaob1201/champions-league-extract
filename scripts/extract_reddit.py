import praw
import pandas as pd
import datetime
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Reddit API credentials
client_id = os.getenv('REDDIT_CLIENT_ID')
client_secret = os.getenv('REDDIT_CLIENT_SECRET')
user_agent = "USER"


def initialize_reddit_client():
    """Initialize and return a Reddit API client."""
    reddit = praw.Reddit(
        client_id=client_id,
        client_secret=client_secret,
        user_agent=user_agent
    )
    return reddit


def search_champions_league_posts(reddit, subreddits=['soccer'], query='champions league', limit=100,
                                  time_filter='month'):
    """Search for Champions League related posts in specified subreddits."""
    posts_data = []

    for subreddit_name in subreddits:
        subreddit = reddit.subreddit(subreddit_name)

        # Search for posts with the query
        for post in subreddit.search(query, sort='relevance', time_filter=time_filter, limit=limit):
            post_data = {
                'id': post.id,
                'title': post.title,
                'score': post.score,
                'url': post.url,
                'created_utc': datetime.datetime.fromtimestamp(post.created_utc),
                'num_comments': post.num_comments,
                'author': str(post.author),
                'selftext': post.selftext,
                'subreddit': subreddit_name,
                'permalink': f'https://www.reddit.com{post.permalink}'
            }
            posts_data.append(post_data)

    return pd.DataFrame(posts_data)


def extract_comments(reddit, post_id, limit=100):
    """Extract comments from a specific post."""
    comments_data = []
    submission = reddit.submission(id=post_id)

    # Replace MoreComments objects with actual comments
    submission.comments.replace_more(limit=None)

    for comment in submission.comments.list()[:limit]:
        comment_data = {
            'id': comment.id,
            'post_id': post_id,
            'body': comment.body,
            'score': comment.score,
            'created_utc': datetime.datetime.fromtimestamp(comment.created_utc),
            'author': str(comment.author),
            'parent_id': comment.parent_id
        }
        comments_data.append(comment_data)

    return pd.DataFrame(comments_data)


def main():
    # Initialize Reddit client
    reddit = initialize_reddit_client()

    # Define parameters
    subreddits = ['soccer', 'championsleague', 'football']
    query = 'champions league'
    limit = 200
    time_filter = 'month'  # Options: hour, day, week, month, year, all

    # Extract posts
    print(f"Searching for '{query}' posts in {subreddits}...")
    posts_df = search_champions_league_posts(reddit, subreddits, query, limit, time_filter)
    print(f"Found {len(posts_df)} posts")

    # Save posts to CSV
    posts_df.to_csv('champions_league_posts.csv', index=False)
    print("Saved posts to champions_league_posts.csv")

    # Extract comments from top posts (optional)
    if not posts_df.empty:
        top_posts = posts_df.sort_values('score', ascending=False).head(10)
        all_comments = []

        for idx, post in top_posts.iterrows():
            print(f"Extracting comments from post: {post['title'][:50]}...")
            comments = extract_comments(reddit, post['id'])
            all_comments.append(comments)

        if all_comments:
            comments_df = pd.concat(all_comments)
            comments_df.to_csv('champions_league_comments.csv', index=False)
            print("Saved comments to champions_league_comments.csv")


if __name__ == "__main__":
    main()