import streamlit as st

st.set_page_config(
      page_title="NLP Hub",
      page_icon="🧊",
)

st.title("Natural Language Processing Hub")
st.sidebar.success("Select a page above.")

st.subheader("",divider='rainbow')

st.image("data/NLP_tasks.png", caption="Source: https://www.nlplanet.org/")
st.markdown("""
Welcome to the NLP Hub! Our platform is dedicated to exploring the vast and intricate landscape of Natural Language Processing (NLP). NLP is a field at the intersection of computer science, artificial intelligence, and linguistics, focusing on the interaction between computers and human languages.

The diagram above, sourced from [NLP Planet](https://www.nlplanet.org/), illustrates various NLP tasks. Here, we delve into key functions essential for understanding and leveraging NLP:
""")

st.markdown("""
* **Text Analysis**: This involves examining the structure and meaning of textual data. Understanding syntax and semantics is crucial for deeper insights. Learn more about text analysis in Steven Bird's book *Natural Language Processing with Python* and [this guide](https://neptune.ai/blog/tokenization-in-nlp).
* **Sentiment Analysis**: Categorizing text based on sentiment (positive, negative, neutral) is essential for understanding public opinion and customer feedback. For a deeper dive, check out [this detailed article](https://www.sketchengine.eu/penn-treebank-tagset/).
* **Text Similarity**: Measuring how similar two pieces of text are can be useful in plagiarism detection and document clustering. Explore different algorithms in [this informative post](https://www.newscatcherapi.com/blog/ultimate-guide-to-text-similarity-with-python#toc-3).
* **Keyword Extraction**: Identifying significant words and phrases helps in indexing and summarization. Understand the methods in [this tutorial](https://towardsdatascience.com/keyword-extraction-process-in-python-with-natural-language-processing-nlp-d769a9069d5c).
* **Topic Modeling**: Discovering hidden themes within large text datasets is invaluable for organizing and understanding information. Learn about topic modeling [here](https://www.datacamp.com/tutorial/what-is-topic-modeling).
* **Text Summarization**: Automatically generating concise summaries makes large volumes of text more digestible. Read more on text summarization techniques in [this article](https://www.geeksforgeeks.org/text-summarization-in-nlp/).
* **Language Translation**: Translating text between languages breaks down communication barriers. For insights into state-of-the-art models, see [this resource](https://medium.com/analytics-vidhya/how-to-translate-text-with-python-9d203139dcf5).
""")

st.markdown("""
Each of these functions is explored in-depth on our platform, with dedicated pages for each topic:
""")

st.markdown('''
    :bar_chart: **Text Analysis**: Techniques and tools for analyzing textual data.  
    :smiley: **Sentiment Analysis**: Training models to understand and classify sentiments in text.  
    :chart_with_upwards_trend: **Text Similarity**: Algorithms for measuring text similarity.  
    :mag_right: **Keyword Extraction**: Methods to extract important keywords from documents.  
    :books: **Topic Modeling**: Techniques to uncover hidden topics within large text corpora.  
    :scissors: **Text Summarization**: Creating summaries that capture the essence of lengthy documents.  
    :globe_with_meridians: **Language Translation**: Translating text between languages using advanced models.
''')

st.markdown("""
Our mission is to **demystify NLP** and equip you with practical tools and insights. Whether you're just starting out or looking for advanced techniques, our platform offers valuable resources to enhance your understanding and application of NLP. Dive into each section and unlock the full potential of Natural Language Processing to transform your interaction with textual data.
""")

st.markdown("""
### References
- Bird, Steven, Edward Loper and Ewan Klein (2009). *Natural Language Processing with Python*. O'Reilly Media Inc.
- [Tokenization in NLP](https://neptune.ai/blog/tokenization-in-nlp)
- [Penn Treebank Tagset](https://www.sketchengine.eu/penn-treebank-tagset/)
- [Ultimate Guide to Text Similarity with Python](https://www.newscatcherapi.com/blog/ultimate-guide-to-text-similarity-with-python#toc-3)
- [Keyword Extraction Process in Python](https://towardsdatascience.com/keyword-extraction-process-in-python-with-natural-language-processing-nlp-d769a9069d5c)
- [What is Topic Modeling?](https://www.datacamp.com/tutorial/what-is-topic-modeling)
- [Text Summarization in NLP](https://www.geeksforgeeks.org/text-summarization-in-nlp/)
- [How to Translate Text with Python](https://medium.com/analytics-vidhya/how-to-translate-text-with-python-9d203139dcf5)
""")
