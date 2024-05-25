import streamlit as st
from gensim.summarization import summarize

st.title("Text Summarization with Gensim")
st.markdown("""
## Text Summarization Page

This page allows you to input text and generate a summary using the Gensim library.
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

# Slider for summary ratio
ratio = st.slider("Summary ratio (fraction of original text):", min_value=0.1, max_value=0.9, value=0.2, step=0.1)

# Submit button
if st.button('Summarize Text'):
    with st.spinner("Summarizing..."):
        try:
            summary = summarize(text, ratio=ratio)
            st.subheader("Original Text")
            st.write(text)
            st.write(f"Original Document Size: {len(text)} characters")

            st.subheader("Summarized Text")
            st.write(summary)
            st.write(f"Summary Length: {len(summary)} characters")
        except ValueError as e:
            st.error(f"Error in summarization: {e}")

# Add the mathematical explanation
st.markdown("""
### Explanation of Summarization

Text summarization is the process of creating a short and coherent version of a longer document. The Gensim library uses a variation of the TextRank algorithm to perform summarization. Here’s a brief explanation of how it works:

1. **Sentence Tokenization**: The input text is split into sentences.
2. **Graph Construction**: Sentences are represented as nodes in a graph.
3. **Edge Weights**: An edge is added between two nodes if they share common words, with the weight of the edge reflecting the degree of similarity.
4. **Ranking Sentences**: Sentences are ranked based on their connections using the PageRank algorithm.
5. **Selection**: The top-ranked sentences are selected to form the summary.

This approach leverages the structure of the text to identify the most important sentences, producing a coherent and concise summary.
""")