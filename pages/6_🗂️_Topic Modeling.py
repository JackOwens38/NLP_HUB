import streamlit as st
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.decomposition import LatentDirichletAllocation, TruncatedSVD
import pandas as pd

# Sample clean corpus
clean_corpus = [
    'artificial intelligence and machine learning are transforming healthcare',
    'predictive analytics and personalized medicine are becoming more prevalent',
    'efficient diagnostic tools and operational efficiencies are improving patient outcomes',
    'healthcare systems are increasingly adopting artificial intelligence for better performance'
]

# Function to print topics
def print_topics(model, vectorizer, top_n=10):
    topics = []
    for idx, topic in enumerate(model.components_):
        topics.append([(vectorizer.get_feature_names_out()[i], topic[i]) for i in topic.argsort()[:-top_n - 1:-1]])
    return topics

st.title("Topic Modeling with LDA and LSA")
st.markdown("""
## Topic Modeling Page

This page allows you to input a corpus, perform topic modeling using LDA (Latent Dirichlet Allocation) and LSA (Latent Semantic Analysis), and view the extracted topics and document-topic distributions.
""")

# Text area for inputting the corpus
text_area_input = st.text_area("Enter documents (one per line):", "\n".join(clean_corpus))

# Split the input text into a list of documents
corpus = [doc.strip() for doc in text_area_input.split('\n') if doc.strip()]

# Number of topics input
num_topics = st.number_input("Enter the number of topics:", min_value=1, max_value=10, value=2)

# Submit button
if st.button('Perform Topic Modeling'):
    if corpus:
        # LDA
        vectorizer_lda = CountVectorizer(stop_words='english')
        doc_term_matrix_lda = vectorizer_lda.fit_transform(corpus)

        lda_model = LatentDirichletAllocation(n_components=num_topics, random_state=42)
        lda_model.fit(doc_term_matrix_lda)

        st.subheader("LDA Topics")
        lda_topics = print_topics(lda_model, vectorizer_lda)
        for idx, topic in enumerate(lda_topics):
            st.write(f"Topic {idx + 1}:")
            for word, weight in topic:
                st.write(f"{word}: {weight:.4f}")

        # LSA
        vectorizer_lsa = TfidfVectorizer(stop_words='english')
        tfidf_matrix_lsa = vectorizer_lsa.fit_transform(corpus)

        lsa_model = TruncatedSVD(n_components=num_topics, random_state=42)
        lsa_model.fit(tfidf_matrix_lsa)

        st.subheader("LSA Topics")
        lsa_topics = print_topics(lsa_model, vectorizer_lsa)
        for idx, topic in enumerate(lsa_topics):
            st.write(f"Topic {idx + 1}:")
            for word, weight in topic:
                st.write(f"{word}: {weight:.4f}")

        # Document-Topic Distributions for LDA
        doc_topic_distributions_lda = lda_model.transform(doc_term_matrix_lda)
        st.subheader("LDA Document-Topic Distributions")
        st.dataframe(pd.DataFrame(doc_topic_distributions_lda, columns=[f"Topic {i + 1}" for i in range(num_topics)]))

        # Document-Topic Distributions for LSA
        doc_topic_distributions_lsa = lsa_model.transform(tfidf_matrix_lsa)
        st.subheader("LSA Document-Topic Distributions")
        st.dataframe(pd.DataFrame(doc_topic_distributions_lsa, columns=[f"Topic {i + 1}" for i in range(num_topics)]))

# Add the mathematical explanation
st.markdown("""
### Mathematical Explanation of LDA and LSA

**Latent Dirichlet Allocation (LDA)**:
LDA is a generative probabilistic model for collections of discrete data such as text corpora. It is based on the idea that documents are mixtures of topics, and topics are mixtures of words.

- **Dirichlet Distribution**: The Dirichlet distribution is used as a prior distribution for the topic proportions in each document.
- **Generative Process**:
  1. For each document, choose a topic distribution from a Dirichlet distribution.
  2. For each word in the document, choose a topic from the topic distribution.
  3. Choose a word from the corresponding topic.

**Latent Semantic Analysis (LSA)**:
LSA is a technique that uses singular value decomposition (SVD) to reduce the dimensionality of a term-document matrix. It captures the underlying structure in the data by identifying patterns in the relationships between terms and documents.

- **Term-Document Matrix**: A matrix representation of the corpus where each row represents a document and each column represents a term.
- **Singular Value Decomposition (SVD)**: SVD factorizes the term-document matrix into three matrices: \( U \), \( \Sigma \), and \( V^T \). The columns of \( U \) represent topics, and the rows of \( V^T \) represent terms.
- **Reduced Dimensionality**: By keeping only the top \( k \) singular values, we reduce the dimensionality of the matrix while preserving the most important information.

Both LDA and LSA are powerful tools for discovering hidden topics in text data and understanding the structure of large corpora.
""")