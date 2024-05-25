import streamlit as st
from sumy.parsers.plaintext import PlaintextParser
from sumy.nlp.tokenizers import Tokenizer
from sumy.summarizers.lsa import LsaSummarizer

st.title("Text Summarization with Sumy")
st.markdown("""
## Text Summarization Page

This page allows you to input text and generate a summary using the Sumy library.
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
num_sentences = st.slider("Number of sentences in summary:", min_value=1, max_value=10, value=2)

# Submit button
if st.button('Summarize Text'):
    parser = PlaintextParser.from_string(text, Tokenizer("english"))
    summarizer = LsaSummarizer()
    summary = summarizer(parser.document, num_sentences)

    summary_text = " ".join(str(sentence) for sentence in summary)

    st.subheader("Original Text")
    st.write(text)
    st.write(f"Original Document Size: {len(text)} characters")

    st.subheader("Summarized Text")
    st.write(summary_text)
    st.write(f"Summary Length: {len(summary_text)} characters")

# Add the mathematical explanation
st.markdown("""
### Explanation of LSA Summarization

Latent Semantic Analysis (LSA) is a technique in natural language processing for analyzing relationships between a set of documents and the terms they contain. Here’s a brief explanation of how it works:

1. **Text Processing**: The text is tokenized and converted into a matrix of term frequencies.
2. **Singular Value Decomposition (SVD)**: The term-document matrix is decomposed into singular values.
3. **Topic Identification**: The topics are identified based on the decomposed matrix.
4. **Sentence Scoring**: Sentences are scored based on how well they represent the identified topics.
5. **Selection**: The top-ranked sentences are selected to form the summary.

LSA leverages the relationships between terms and documents to identify the most important sentences, producing a coherent and concise summary.
""")