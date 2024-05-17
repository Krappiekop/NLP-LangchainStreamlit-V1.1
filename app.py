import streamlit as st
from langchain_community.llms import OpenAI as OpenAIModel
import spacy
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from textblob import TextBlob
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, AutoModelForSequenceClassification
import torch
import nltk



st.title('NLP Application') #Title

# Download NLP model
nltk.download('wordnet')
# Load NLP models
spacy.download("en_core_web_sm")
nlp = spacy.load("en_core_web_sm")

# Load models for summarization and sentiment analysis
summarization_model_name = "facebook/bart-large-cnn"
tokenizer_summarization = AutoTokenizer.from_pretrained(summarization_model_name)
model_summarization = AutoModelForSeq2SeqLM.from_pretrained(summarization_model_name)

sentiment_model_name = "nlptown/bert-base-multilingual-uncased-sentiment"
tokenizer_sentiment = AutoTokenizer.from_pretrained(sentiment_model_name)
model_sentiment = AutoModelForSequenceClassification.from_pretrained(sentiment_model_name)

# Function to handle ChatGPT interaction
def chat_with_gpt():
    st.title('Chat with AI')
    openai_api_key = st.sidebar.text_input('OpenAI API Key')

    def generate_response(input_text):
        llm = OpenAIModel(temperature=0.7, openai_api_key=openai_api_key)
        response = llm(input_text)
        st.info(response)

    with st.form('my_form'):
        text = st.text_area('Enter text:', 'What are the three key pieces of advice for learning how to code?')
        submitted = st.form_submit_button('Submit')
        if not openai_api_key.startswith('sk-'):
            st.warning('Please enter your OpenAI API key!', icon='⚠️')
        if submitted and openai_api_key.startswith('sk-'):
            generate_response(text)

# NLP functions
def nlp_summary(text):
    inputs = tokenizer_summarization.encode("summarize: " + text, return_tensors="pt", max_length=512, truncation=True)
    summary_ids = model_summarization.generate(inputs, max_length=150, min_length=40, length_penalty=2.0, num_beams=4, early_stopping=True)
    summary = tokenizer_summarization.decode(summary_ids[0], skip_special_tokens=True)
    return summary

def nlp_sentiment(text):
    inputs = tokenizer_sentiment(text, return_tensors="pt", truncation=True)
    outputs = model_sentiment(**inputs)
    probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
    sentiment = torch.argmax(probs, dim=1).item()
    sentiment_labels = ['very negative', 'negative', 'neutral', 'positive', 'very positive']
    return sentiment_labels[sentiment]

def nlp_ner(text):
    doc = nlp(text)
    entities = [(ent.text, ent.label_) for ent in doc.ents]
    return entities

def nlp_tokenize(text):
    tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()
    tokens_and_lemmas = [(token, lemmatizer.lemmatize(token)) for token in tokens]
    return tokens_and_lemmas

# Function to handle NLP tasks
def nlp_tasks():
    st.title('Process Text')
    st.subheader('Natural Language Processing for everyone')
    st.write("""
        This is a Natural Language Processing (NLP) Based App useful for basic NLP tasks: Tokenization, Lemmatization, 
        Named Entity Recognition (NER), Sentiment Analysis, Text Summarization.
        Click any of the checkboxes to get started.
    """)

    input_text = st.text_area("Enter your text here:")

    if st.checkbox("Get the summary of your text"):
        summary = nlp_summary(input_text)
        st.write("Summary:")
        st.info(summary)

    if st.checkbox("Get the Sentiment Score of your text"):
        sentiment = nlp_sentiment(input_text)
        st.write("Sentiment Score:")
        st.info(sentiment)

    if st.checkbox("Get the Named Entities of your text"):
        entities = nlp_ner(input_text)
        st.write("Named Entities:")
        st.info(entities)

    if st.checkbox("Get the Tokens and Lemma of text"):
        tokens = nlp_tokenize(input_text)
        st.write("Tokens and Lemma:")
        st.info(tokens)

# Add sidebar for navigation
st.sidebar.title('Navigation')
options = st.sidebar.radio('Go to', ['Chat with GPT', 'NLP Tasks'])

if options == 'Chat with GPT':
    chat_with_gpt()
elif options == 'NLP Tasks':
    nlp_tasks()
