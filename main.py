import nltk
import gensim
from gensim import corpora
from transformers import T5Tokenizer, T5ForConditionalGeneration, set_seed

# Set random seed for reproducibility
set_seed(42)

# Load the input text
input_text = "This is some sample input text. It has multiple sentences, and we want to extract topics from it using LDA."

# Tokenize the input text
sentences = nltk.sent_tokenize(input_text)
tokenized_sentences = [nltk.word_tokenize(sentence) for sentence in sentences]

# Build a dictionary from the tokenized sentences
dictionary = corpora.Dictionary(tokenized_sentences)

# Convert the tokenized sentences to a bag-of-words representation
corpus = [dictionary.doc2bow(sentence) for sentence in tokenized_sentences]

# Train an LDA model on the corpus
lda_model = gensim.models.ldamodel.LdaModel(corpus, num_topics=3, id2word=dictionary, passes=10)

# Extract the topics from the LDA model
topics = []
for topic_id in range(lda_model.num_topics):
    topic_words = lda_model.show_topic(topic_id, topn=10)
    topics.append([word for word, _ in topic_words])

# Generate topic-wise summaries using T5ForConditionalGeneration
tokenizer = T5Tokenizer.from_pretrained("t5-small")
model = T5ForConditionalGeneration.from_pretrained("t5-small")

summaries_by_topic = {}
for topic in topics:
    # Filter the sentences in the input text that contain the topic words
    topic_sentences = []
    for sentence in sentences:
        if any(word in sentence.lower() for word in topic):
            topic_sentences.append(sentence)

    # Generate a summary for the topic
    if len(topic_sentences) > 0:
        topic_text = ' '.join(topic_sentences)
        input_ids = tokenizer.encode("summarize: " + topic_text, return_tensors="pt", max_length=512, truncation=True)
        summary_ids = model.generate(input_ids, num_beams=4, no_repeat_ngram_size=2, min_length=30, max_length=100)
        summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
        summaries_by_topic[' '.join(topic)] = summary

# Print the summaries for all topics
for topic, summary in summaries_by_topic.items():
    print(f"Topic: {topic}")
    print(f"Summary: {summary}")
    print("="*50)
