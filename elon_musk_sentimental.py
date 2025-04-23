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