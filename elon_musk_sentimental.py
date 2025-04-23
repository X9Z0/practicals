import pandas as pd
from textblob import TextBlob

# Load the dataset
df = pd.read_csv('/mnt/data/ElonMusk_tweets.csv')

# Check the column names
print(df.columns)

# Assuming the tweets are in a column named 'content' or similar
# Replace 'content' with the correct column name if different
df['polarity'] = df['content'].apply(lambda x: TextBlob(str(x)).sentiment.polarity)
df['subjectivity'] = df['content'].apply(lambda x: TextBlob(str(x)).sentiment.subjectivity)

# Classify sentiment based on polarity
def get_sentiment(p):
    if p > 0:
        return 'Positive'
    elif p < 0:
        return 'Negative'
    else:
        return 'Neutral'

df['sentiment'] = df['polarity'].apply(get_sentiment)

# View the result
print(df[['content', 'polarity', 'subjectivity', 'sentiment']].head())

from textblob import TextBlob
import pandas as pd

def analyze_sentiment(tweets):
    """
    tweets: A list of tweet strings or a pandas DataFrame with a 'content' column.
    Returns: A DataFrame with sentiment analysis results.
    """
    if isinstance(tweets, list):
        df = pd.DataFrame(tweets, columns=['content'])
    elif isinstance(tweets, pd.DataFrame) and 'content' in tweets.columns:
        df = tweets.copy()
    else:
        raise ValueError("Input must be a list of tweets or a DataFrame with a 'content' column.")

    df['polarity'] = df['content'].apply(lambda x: TextBlob(str(x)).sentiment.polarity)
    df['subjectivity'] = df['content'].apply(lambda x: TextBlob(str(x)).sentiment.subjectivity)

    def get_sentiment(p):
        if p > 0:
            return 'Positive'
        elif p < 0:
            return 'Negative'
        else:
            return 'Neutral'

    df['sentiment'] = df['polarity'].apply(get_sentiment)
    return df[['content', 'polarity', 'subjectivity', 'sentiment']]

new_tweets = [
    "Tesla stock is soaring again!",
    "I'm not sure about the future of cryptocurrencies.",
    "SpaceX launch was amazing!",
    "The new update is terrible."
]

results = analyze_sentiment(new_tweets)
print(results)