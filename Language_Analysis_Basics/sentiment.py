from nltk.sentiment import SentimentIntensityAnalyzer # vader sentiment analysis tool

turn_exchange = "Oh, god, shoot. Wait, um"

sia = SentimentIntensityAnalyzer()
sentiment_score = sia.polarity_scores(turn_exchange)

print(sentiment_score)
