import stanza
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

# Download the English models for the neural pipeline
# stanza.download('en')

# Initialize the Stanza pipeline
nlp = stanza.Pipeline('en', processors='tokenize,sentiment')

# Initialize VADER sentiment analyzer
vader_analyzer = SentimentIntensityAnalyzer()

# Define the dialogue
dialogue = [
    "Hi can you help me find a movie to watch",
    "Yes, how about @187028 <It  (2017)>?",
    "I  dislike horror movies what about thrillers",
    "@203424 <The Silence of the Lambs  (1991)>",
    "I have seen it and enjoyed it...One of my favorite thrillers is @89770 <The Number 23 (2007)> and @170119 <The Sixth Sense (1999)>",
    "I have not seen @89770 <The Number 23 (2007)>, but I did enjoy @170119 <The Sixth Sense (1999)>...I'm more into horror movies though.",
    "That is okay I just do not really like the @163704 <Halloween (2007)> movies",
    "That's literally my favorite movie series hahaha....Like ever",
    "That is a lot of peoples favorite I just had a bad experience with the movie. A horror movie will be ok I did enjoy @125431 <Annabelle (2014)> and @146741 <Curse of Chucky (2013)>",
    "I like @127513 <Halloween: Resurrection (2002)>...I like the original @143062 <Child's Play (1988)>",
    "I have seen that one and enjoyed it. Is there any you can recommend like that movie",
    "I also like @185171 <The Conjuring (2013)> where @125431 <Annabelle (2014)> came from....@100454 <Dolls (1987)>",
    "I think I will check out @185171 <The Conjuring (2013)> thanks for the help",
    "Ok have a good day",
    "Thank you bye."
]

# Analyze sentiment of each utterance
for idx, utterance in enumerate(dialogue):
    # Stanza sentiment analysis
    doc = nlp(utterance)
    stanza_sentiment = doc.sentences[0].sentiment
    if stanza_sentiment == 0:
        stanza_label = 'Negative'
    elif stanza_sentiment == 1:
        stanza_label = 'Neutral'
    else:
        stanza_label = 'Positive'

    # VADER sentiment analysis
    vader_scores = vader_analyzer.polarity_scores(utterance)
    vader_compound = vader_scores['compound']
    if vader_compound <= -0.6:
        vader_label = 'Very Negative'
    elif -0.6 < vader_compound <= -0.2:
        vader_label = 'Negative'
    elif -0.2 < vader_compound <= 0.2:
        vader_label = 'Neutral'
    elif 0.2 < vader_compound <= 0.6:
        vader_label = 'Positive'
    else:
        vader_label = 'Very Positive'

    print(f"Utterance {idx + 1}: {utterance}")
    print(f"Stanza Sentiment: {stanza_label}")
    print(f"VADER Sentiment: {vader_label}\n")
