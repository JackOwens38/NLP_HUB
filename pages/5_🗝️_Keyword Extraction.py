import streamlit as st
from rake_nltk import Rake
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import time

# Function to generate word cloud
def generate_word_cloud(text):
    wordcloud = WordCloud(width=800, height=400, background_color='white').generate(text)
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.imshow(wordcloud, interpolation='bilinear')
    ax.axis('off')
    return fig

st.title("Keyword Extraction and Word Cloud Generator")
st.markdown("""
## Keyword Extraction and Word Cloud

This page allows you to input text, extract keywords using RAKE, and generate a word cloud based on the text.
""")

# Text input
text = st.text_area("Enter text for keyword extraction and word cloud generation:", "Artificial intelligence and machine learning are transforming healthcare by enabling predictive analytics, personalized medicine, and efficient diagnostic tools, which significantly improve patient outcomes and operational efficiencies.")

# Submit button
if st.button('Extract Keywords and Generate Word Cloud'):
    if text:
        with st.spinner('Extracting...'):
            time.sleep(3)
            # Extract keywords using RAKE
            rake = Rake()
            rake.extract_keywords_from_text(text)
            keywords = rake.get_ranked_phrases_with_scores()

            st.subheader("Extracted Keywords and Scores")
            for score, phrase in keywords:
                st.write(f"{score}: {phrase}")

            st.subheader("Word Cloud")
            # Generate and display the word cloud
            fig = generate_word_cloud(text)
            st.pyplot(fig)

# Add the mathematical explanation
st.markdown("""
### Mathematical Explanation of RAKE

RAKE (Rapid Automatic Keyword Extraction) is a keyword extraction algorithm that identifies key phrases in a body of text. Here’s a brief explanation of how it works:

1. **Tokenization**: The text is split into words.
2. **Phrase Extraction**: Sequences of words that do not contain stop words are extracted as candidate keywords.
3. **Word Scores**: Each word is scored based on its frequency and degree (number of times it appears in candidate keywords).
4. **Phrase Scores**: Candidate keywords are scored by summing the scores of their constituent words.

The score of a word \( w \) is calculated as:
""")
st.latex(r'''
\text{Score}(w) = \text{Frequency}(w) + \text{Degree}(w)
''')
st.latex(r'''
\text{Where:}
\begin{align*}
\text{Frequency}(w) & \text{ is the number of occurrences of the word.} \\
\text{Degree}(w) & \text{ is the sum of the co-occurrences of the word with other words.}
\end{align*}
''')
st.markdown("""
The score of a phrase \( P \) is the sum of the scores of the words in the phrase.
""")