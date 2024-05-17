import streamlit as st
from langchain.llms import OpenAI
from langchain import OpenAI as OpenAIModel

st.title('NLP Application')

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

# Function to handle NLP tasks
def nlp_tasks():
    st.title('Process Text')
    st.subheader('Natural Language Processing for everyone')
    st.write("""
        This is a Natural Language Processing(NLP) Based App useful for basic NLP tasks: Tokenization, Lemmatization, 
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

# NLP function implementations (placeholders for actual implementation)
def nlp_summary(text):
    return "This is a summary of the text."

def nlp_sentiment(text):
    return "This is the sentiment score of the text."

def nlp_ner(text):
    return "These are the named entities in the text."

def nlp_tokenize(text):
    return "These are the tokens and lemma of the text."

# Main script
st.sidebar.title("Navigation")
page = st.sidebar.selectbox("Go to", ["Chat with GPT", "NLP Tasks"])

if page == "Chat with GPT":
    chat_with_gpt()
else:
    nlp_tasks()
