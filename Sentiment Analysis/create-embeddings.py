import os
import jsonlines
import stanza
from sentence_transformers import SentenceTransformer

# Disable the symlink warning
os.environ['HF_HUB_DISABLE_SYMLINKS_WARNING'] = '1'

# Initialize Stanza pipeline for sentiment analysis
stanza.download('en')  # Ensure the models are downloaded
nlp = stanza.Pipeline(lang='en', processors='tokenize,sentiment')

# Initialize Sentence-Transformer model for embeddings
embedder = SentenceTransformer('all-MiniLM-L6-v2')

# Function to map sentiment scores to labels
def map_sentiment(score):
    if score == 0:
        return "negative"
    elif score == 1:
        return "neutral"
    elif score == 2:
        return "positive"

# Function to analyze sentiments for a single conversation
def analyze_conversation(conversation):
    movie_sentiments = {}
    movie_mentions = conversation.get("movieMentions", {})
    messages = conversation.get("messages", [])

    current_movie_id = None
    current_texts = []

    def add_movie_sentiments(movie_id, texts):
        combined_text = ". ".join(texts) + "."
        doc = nlp(combined_text)
        sentiment_scores = [sentence.sentiment for sentence in doc.sentences]
        avg_sentiment_score = sum(sentiment_scores) / len(sentiment_scores)
        sentiment_label = map_sentiment(round(avg_sentiment_score))
        
        # Convert sentiment label to embedding
        sentiment_embedding = embedder.encode(combined_text)
        
        if movie_id not in movie_sentiments:
            movie_sentiments[movie_id] = []
        movie_sentiments[movie_id].append((sentiment_label, sentiment_embedding, combined_text))

    for message in messages:
        found_new_movie = False
        for movie_id in movie_mentions.keys():
            if movie_id in message["text"]:
                if current_movie_id and current_movie_id != movie_id:
                    add_movie_sentiments(current_movie_id, current_texts)
                current_movie_id = movie_id
                current_texts = [message["text"]]
                found_new_movie = True
                break
        if not found_new_movie and current_movie_id:
            current_texts.append(message["text"])

    if current_movie_id:
        add_movie_sentiments(current_movie_id, current_texts)

    return movie_sentiments

# Load and analyze the dataset
movie_sentiments = {}

with jsonlines.open('short_data.jsonl') as file:
    for conversation in file:
        conversation_sentiments = analyze_conversation(conversation)

        for movie_id, sentiment_info in conversation_sentiments.items():
            if movie_id not in movie_sentiments:
                movie_sentiments[movie_id] = []
            movie_sentiments[movie_id].extend(sentiment_info)

# Print the results
for movie_id, sentiments in movie_sentiments.items():
    print(f"Movie ID: {movie_id}")
    for sentiment, embedding, sentence in sentiments:
        print(f"  Sentiment: {sentiment}, Embedding: {embedding[:5]}..., Sentence: {sentence}")