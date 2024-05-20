import streamlit as st


st.set_page_config(
      page_title="NLP Hub",
      page_icon="🧊",
)

st.title("Natural Language Processing Hub")
st.sidebar.success("Select a page above.")

if "my_input" not in st.session_state:
    st.session_state.my_input = ""

my_input = st.text_input("Input a text here", st.session_state["my_input"])
submit = st.button("Submit")
if submit:
    st.session_state["my_input"]= my_input
    st.write("You have entered: ", my_input)

    
st.subheader("",divider='rainbow')


st.image("data/NLP_tasks.png", caption="Source: https://www.nlplanet.org/")
st.markdown("""
The diagram above illustrates the vast and intricate landscape of Natural Language Processing (NLP). NLP encompasses a variety of tasks and technologies that enable computers to understand, interpret, and generate human language. Our application focuses on several key functions that are essential in NLP:
""")

st.markdown("""
* **Text Analysis**: Dive into the nuances of textual data, exploring its structure and meaning.
* **Text Classification**: Categorize text into predefined labels, an essential function for sentiment analysis and spam detection.
* **Text Similarity**: Measure the similarity between text documents, enabling applications like plagiarism detection and document clustering.
* **Keyword Extraction**: Identify the most significant words and phrases within a text, useful for indexing and summarization.
* **Topic Modeling**: Discover the hidden themes within large volumes of text, aiding in the organization and understanding of extensive datasets.
* **Text Summarization**: Automatically generate concise summaries of documents, making it easier to digest large amounts of information.
* **Language Translation**: Break down language barriers by translating text between different languages with high accuracy.
""")

st.markdown("""
Our application is structured to provide an in-depth understanding of each of these functions, with separate pages dedicated to exploring them:
""")

st.markdown('''
    :bar_chart: **Text Analysis**: Learn about the techniques and tools used to analyze textual data.  
    :label: **Text Classification**: Understand how to train models to classify text into different categories.  
    :chart_with_upwards_trend: **Text Similarity**: Explore algorithms that measure how similar two pieces of text are.  
    :mag_right: **Keyword Extraction**: Discover methods to extract important keywords from documents.  
    :books: **Topic Modeling**: Uncover hidden topics within large text corpora.  
    :scissors: **Text Summarization**: Create summaries that capture the essence of lengthy documents.  
    :globe_with_meridians: **Language Translation**: Translate text between languages with state-of-the-art models.
''')

st.markdown("""
Our goal is to **demystify NLP** and provide you with practical tools and insights to leverage these powerful technologies. Whether you are a beginner looking to understand the basics or an expert seeking advanced techniques, our application offers something for everyone. Explore each section to unlock the full potential of Natural Language Processing and transform the way you interact with textual data.
""")


