import streamlit as st
from deep_translator import GoogleTranslator
import time
st.title("Language Translation with Google Translator")
st.markdown("""
## Language Translation Page

This page allows you to input text, select source and target languages, and view the translated text using the Google Translator API.
""")

# Text area for inputting the text
text = st.text_area("Enter text to translate:", "I like to live my life.")

# Select box for source language
source_lang = st.selectbox("Select the source language:", ['auto', 'en', 'fr', 'de', 'es', 'it', 'zh', 'ja', 'ko', 'ar'])

# Select box for target language
target_lang = st.selectbox("Select the target language:", ['ar', 'en', 'fr', 'de', 'es', 'it', 'zh', 'ja', 'ko'])

# Submit button
if st.button('Translate'):
    with st.spinner('🌐...'):
        time.sleep(3)

    if text:
        # Translate the text using GoogleTranslator
        translated = GoogleTranslator(source=source_lang, target=target_lang).translate(text)
        
        st.subheader("Original Text")
        st.write(text)
        
        st.subheader("Translated Text")
        st.write(translated)

# Add the explanation
st.markdown("""
### Explanation

The `deep_translator` library provides a convenient way to use the Google Translator API for translating text between various languages. Here’s how it works:

1. **Input Text**: The text to be translated is provided by the user.
2. **Language Selection**: The user selects the source and target languages.
3. **Translation**: The Google Translator API is called to translate the text from the source language to the target language.

The translated text is then displayed to the user.

#### Example Usage

For example, if you input the text "I like to live my life." and select the target language as Arabic (`ar`), the translated text will be displayed in Arabic.
""")