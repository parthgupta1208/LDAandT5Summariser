import nltk
import gensim
from gensim import corpora
from transformers import T5Tokenizer, T5ForConditionalGeneration, set_seed

# Set random seed for reproducibility
set_seed(42)

# Load the input text
input_text = """Introduction:
The Ganges River, one of India's most significant and revered waterways, passes through the city of 
Prayagraj. The purpose of this survey report is to assess the current state of water management in and 
around Prayagraj, with a focus on sustainability. The report aims to gather information on the challenges 
faced by local communities, current water management practices, and potential solutions to improve the 
water management system in the area, particularly for the Ganges River and its tributaries. By 
understanding the challenges and opportunities related to water management in the region, the goal is to 
develop a sustainable approach to water management that benefits local communities, protects the 
environment, and supports the long-term viability of the Ganges River and its ecosystems.
Methodology:
The survey was conducted using both qualitative and quantitative research methods, including in-person 
interviews with local residents, farmers, and government officials, as well as a review of relevant data 
and literature. A total of 50 participants were interviewed, and a questionnaire was administered to gather 
information on their experiences and perspectives on the water management system in the area.
Findings:
• Water scarcity: Participants reported that water scarcity is a major issue in the area, with many 
communities facing water shortages during the dry season.
• Inefficient water usage: Many participants reported that current water management practices are 
inefficient, with much of the water being wasted due to leaks, evaporation, and other factors.
• Lack of proper infrastructure: Participants also reported that the current infrastructure for water 
management, such as wells, reservoirs, and pipelines, is in need of repair or replacement.
• Poor water quality: Some participants reported that the water quality in the area is poor, with high 
levels of pollutants and salinity.
• Lack of community involvement: Many participants noted that there is a lack of community 
involvement in water management, with decisions often being made by government officials 
without consulting local residents.
 (An Autonomous Institution affiliated to VTU, Belagavi)
 Nitte-574 110, Karkala Taluk, Udupi District.
Signature of Student Signature of Department NSS Coordinator
Conclusion:
In conclusion, the survey findings indicate that there are significant challenges to developing a 
sustainable water management system in and around Prayagraj. To address these challenges, it will be 
important to improve the efficiency of water usage, invest in infrastructure upgrades, improve water 
quality, and engage local communities in water management decisions.
Recommendations:
• Implement water conservation measures: This could include measures such as rainwater 
harvesting, efficient irrigation techniques, and reducing water wastage.
• Upgrade infrastructure: This could involve repairing and upgrading existing water management 
infrastructure, such as wells, reservoirs, and pipelines, as well as building new infrastructure 
where necessary.
• Improve water quality: This could involve treating water to remove pollutants and salinity, as 
well as reducing pollution through better waste management practices.
• Engage local communities: It will be important to involve local communities in water 
management decisions, such as through the creation of community-based water management 
committees.
• Increase awareness: Raising awareness about the importance of sustainable water management, 
through education and outreach programs, can help to build support for these initiatives.
• Encourage collaboration: Collaboration between government, NGOs, and private sector 
organizations can help to leverage resources and expertise to address the challenges of water 
management in the area.
• Implement technology: The use of technology, such as remote sensing and data analysis, can help 
to improve water management practices by providing real-time information on water availability, 
usage, and quality.
• Monitor and evaluate: Regular monitoring and evaluation of water management practices can 
help to identify areas for improvement and track progress over time.
• Foster community-based initiatives: Encouraging and supporting community-based initiatives, 
such as water-saving projects, can help to build local capacity and drive sustainable water 
management practices.
• Foster public-private partnerships: Encouraging public-private partnerships can help to bring in 
private sector resources, expertise, and innovative solutions to support water management efforts.
These recommendations are not exhaustive, but they provide a starting point for developing a sustainable 
water management system in and around Prayagraj, particularly for the Ganges River and its tributaries. 
By addressing the challenges related to water management in the region, it will be possible to ensure that 
water resources are managed in a way that meets the needs of local communities, while also preserving 
and protecting these resources for future generations. Furthermore, this will contribute to the 
preservation of the Ganges River and its ecosystem in Prayagraj, which is an essential source of water 
for local communities, agriculture, and wildlife"""
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
