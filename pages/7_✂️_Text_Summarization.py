import streamlit as st
import spacy
import pytextrank
import time
# Load Spacy model and add the TextRank pipeline component
try:
    nlp = spacy.load("en_core_web_lg")
    nlp.add_pipe("textrank")
except OSError:
    st.error("Spacy model 'en_core_web_lg' not found. Please ensure the model is downloaded.")


st.title("Text Summarization with SpaCy and pyTextRank")
st.markdown("""
## Text Summarization Page

This page allows you to input text and generate a summary using the pyTextRank library with SpaCy.
""")

# Sample text
sample_text = """Deep learning (also known as deep structured learning) is part of a 
broader family of machine learning methods based on artificial neural networks with 
representation learning. Learning can be supervised, semi-supervised or unsupervised. 
Deep-learning architectures such as deep neural networks, deep belief networks, deep reinforcement learning, 
recurrent neural networks and convolutional neural networks have been applied to
fields including computer vision, speech recognition, natural language processing, 
machine translation, bioinformatics, drug design, medical image analysis, material
inspection and board game programs, where they have produced results comparable to 
and in some cases surpassing human expert performance. Artificial neural networks
(ANNs) were inspired by information processing and distributed communication nodes
in biological systems. ANNs have various differences from biological brains. Specifically, 
neural networks tend to be static and symbolic, while the biological brain of most living organisms
is dynamic (plastic) and analogue. The adjective "deep" in deep learning refers to the use of multiple
layers in the network. Early work showed that a linear perceptron cannot be a universal classifier, 
but that a network with a nonpolynomial activation function with one hidden layer of unbounded width can.
Deep learning is a modern variation which is concerned with an unbounded number of layers of bounded size, 
which permits practical application and optimized implementation, while retaining theoretical universality 
under mild conditions. In deep learning the layers are also permitted to be heterogeneous and to deviate widely 
from biologically informed connectionist models, for the sake of efficiency, trainability and understandability, 
whence the structured part."""

# Text area for inputting the text
text = st.text_area("Enter text to summarize:", sample_text, height=300)

# Parameters for the summary
limit_phrases = st.slider("Limit phrases:", min_value=1, max_value=10, value=2)
limit_sentences = st.slider("Limit sentences:", min_value=1, max_value=10, value=2)

# Submit button
if st.button('Summarize Text'):

    if text:
        # Process the text with SpaCy and pyTextRank
        doc = nlp(text)
        summary = "\n".join([str(sent) for sent in doc._.textrank.summary(limit_phrases=limit_phrases, limit_sentences=limit_sentences)])

        st.subheader("Original Text")
        st.write(text)
        st.write(f"Original Document Size: {len(text)} characters")

        st.subheader("Summarized Text")
        st.write(summary)
        st.write(f"Summary Length: {len(summary)} characters")

# Add the mathematical explanation
st.markdown("""
### Mathematical Explanation of TextRank

TextRank is an unsupervised text summarization technique based on the PageRank algorithm used in web search engines. Here’s a brief explanation of how it works:

1. **Graph Construction**: Sentences or phrases in the text are represented as nodes in a graph.
2. **Edge Weights**: An edge is added between two nodes if they are similar, with the weight of the edge reflecting the degree of similarity.
3. **Scoring**: Each node is scored based on its connections using the PageRank algorithm:
""")

st.latex(r'''
PR(V_i) = (1 - d) + d \sum_{V_j \in In(V_i)} \frac{PR(V_j)}{L(V_j)}
''')

st.markdown("""
   Where:
   - $ PR(V_i) $ is the PageRank score of node $ V_i $.
   - $ d $ is a damping factor (usually set to 0.85).
   - $ In(V_i) $ is the set of nodes that link to $ V_i $.
   - $ L(V_j) $ is the number of outbound links from node $ V_j $.

4. **Selection**: The top-ranked nodes (sentences or phrases) are selected to form the summary.

TextRank leverages the graph structure of the text to identify the most important sentences, producing a coherent and concise summary.
""")