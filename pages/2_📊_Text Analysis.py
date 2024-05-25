import streamlit as st
import pandas as pd
import nltk
from nltk.tokenize import (word_tokenize, 
                           sent_tokenize, 
                           TreebankWordTokenizer, 
                           wordpunct_tokenize, 
                           TweetTokenizer, 
                           MWETokenizer)
from nltk import download
import re #Remove Punctuation
from nltk.stem import PorterStemmer #Stemming
from nltk.stem import WordNetLemmatizer #Lemmatization
import time

nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
nltk.download('wordnet')





st.title("Text Analysis")
st.markdown("""
## Table of Contents
1. [Tokenization](#tokenization)
2. [Part of Speech (POS) Tagging](#part-of-speech-pos-tagging)
3. [Removal of Punctuation](#removal-of-punctuation)
4. [Lowercasing](#lowercasing)
5. [Stemming](#stemming)
6. [Lemmatization](#lemmatization)
""")

st.markdown("## Tokenization")
st.subheader("",divider='rainbow')
st.markdown("""### What is Tokenization?

Tokenization is the process of breaking down text into smaller units called tokens. These tokens can be words, sentences, or subwords, depending on the type of tokenization. Here, we showcase different tokenization methods available in the NLTK library.

### Types of Tokenization

- **Word Tokenization**: Splits text into individual words.
- **Sentence Tokenization**: Splits text into individual sentences.
- **Treebank Word Tokenization**: Uses the Penn Treebank method to tokenize text, handling punctuation and special characters.
- **WordPunct Tokenization**: Splits text based on word boundaries and punctuation marks.
- **Tweet Tokenization**: Designed specifically for tokenizing tweets and handles hashtags, mentions, and emoticons.
- **Multi-word Expression Tokenization**: Identifies and tokenizes multi-word expressions.
""")


# text box
text = st.text_area("Enter text to tokenize:", "Natural Language Processing is fascinating and very useful. Let's tokenize this text! Also, 3 + 5 = 8.", key="tokenize_text_input")

if st.button('Submit', key='tokenize_submit'):
  with st.spinner("🔍🔎🔍"):
     time.sleep(3)
  if text:
      # Word tokenization
      word_tokens = word_tokenize(text)
    
      # Sentence tokenization
      sentence_tokens = sent_tokenize(text)
    
      # Treebank word tokenization
      treebank_word_tokenizer = TreebankWordTokenizer()
      treebank_word_tokens = treebank_word_tokenizer.tokenize(text)
    
      # Word punct tokenization
      wordpunct_tokens = wordpunct_tokenize(text)
    
      # Tweet tokenization
      tweet_tokenizer = TweetTokenizer()
      tweet_tokens = tweet_tokenizer.tokenize(text)
    
      # Multi-word expression tokenization
      mwe_tokenizer = MWETokenizer([('Natural', 'Language', 'Processing')])
      mwe_tokens = mwe_tokenizer.tokenize(word_tokenize(text))

      # Display the results
      st.subheader("Tokenization Results")

      st.markdown("### Word Tokenization")
      st.write(word_tokens)

      st.markdown("### Sentence Tokenization")
      st.write(sentence_tokens)

      st.markdown("### Treebank Word Tokenization")
      st.write(treebank_word_tokens)

      st.markdown("### WordPunct Tokenization")
      st.write(wordpunct_tokens)

      st.markdown("### Tweet Tokenization")
      st.write(tweet_tokens)

      st.markdown("### Multi-word Expression Tokenization")
      st.write(mwe_tokens)
    
# Explanation of tokenization
st.markdown("""
### Example with Arithmetic Expression

Let's see how these tokenizers handle an arithmetic expression:

- Input: "3 + 5 = 8"
- Word Tokenization: `['3', '+', '5', '=', '8']`
- Sentence Tokenization: `['3 + 5 = 8']`
- Treebank Word Tokenization: `['3', '+', '5', '=', '8']`
- WordPunct Tokenization: `['3', '+', '5', '=', '8']`
- Tweet Tokenization: `['3', '+', '5', '=', '8']`
- Multi-word Expression Tokenization: `['3', '+', '5', '=', '8']`

Tokenization helps in breaking down text into manageable pieces for further processing and analysis in NLP tasks.
""")




st.markdown("## Part of Speech (POS) Tagging")
st.subheader("",divider='rainbow')
st.markdown("""
### Part of Speech (POS) Tagging
POS tagging is a fundamental step in understanding the grammatical structure of a sentence. Each word is tagged with its corresponding part of speech, such as noun, verb, adjective, etc. This information is crucial for various NLP tasks such as parsing, sentiment analysis, and information extraction.

To use POS tagging, simply enter a sentence in the text box below, and the system will tokenize the sentence and display the part of speech tags for each word.
""")








# Input text box
sentence = st.text_input("Enter a sentence:", "Natural Language Processing is very useful.", key="sentence_pos_input")

if st.button('Submit', key='submit_button_pos'):
    with st.spinner("📜🔍📜"):
       time.sleep(3)
    if sentence:
        # Tokenize the sentence
        tokens = TreebankWordTokenizer().tokenize(sentence)
        
        # Perform POS tagging on the tokens
        pos_tags = nltk.pos_tag(tokens)
        
        # Convert POS tags to a pandas DataFrame
        pos_df = pd.DataFrame(pos_tags, columns=["Word", "POS Tag"])
        
        # Display the POS tags in a table format
        st.subheader("POS Tags")
        st.table(pos_df)





st.markdown("""
### Penn Treebank Tagset Overview

A tagset is a list of part-of-speech tags, i.e., labels used to indicate the part of speech and often also other grammatical categories (case, tense etc.) of each token in a text corpus.

#### Introduction to Penn Treebank Tagset

The English Penn Treebank tagset is utilized with English corpora annotated by the TreeTagger tool, developed by Helmut Schmid in the TC project at the Institute for Computational Linguistics of the University of Stuttgart. This version of the tagset contains modifications developed by Sketch Engine (earlier version).

[See a more recent version of this tagset](https://www.sketchengine.eu/penn-treebank-tagset/).

### What is a POS Tag?

POS tags classify words into grammatical categories which can help in understanding the structure and context of text. The table below shows the English Penn TreeBank tagset with Sketch Engine modifications (earlier version).

Example: Using `[tag="NNS"]` finds all nouns in the plural, e.g., people, years when used in the CQL concordance search (always use straight double quotation marks in CQL).

| POS Tag | Description                                   | Example             |
|---------|-----------------------------------------------|---------------------|
| CC      | Coordinating conjunction                      | and                 |
| CD      | Cardinal number                               | 1, third            |
| DT      | Determiner                                    | the                 |
| EX      | Existential there                             | there is            |
| FW      | Foreign word                                  | les                 |
| IN      | Preposition, subordinating conjunction        | in, of, like        |
| IN/that | That as subordinator                          | that                |
| JJ      | Adjective                                     | green               |
| JJR     | Adjective, comparative                        | greener             |
| JJS     | Adjective, superlative                        | greenest            |
| LS      | List marker                                   | 1)                  |
| MD      | Modal                                         | could, will         |
| NN      | Noun, singular or mass                        | table               |
| NNS     | Noun, plural                                  | tables              |
| NP      | Proper noun, singular                         | John                |
| NPS     | Proper noun, plural                           | Vikings             |
| PDT     | Predeterminer                                 | both the boys       |
| POS     | Possessive ending                             | friend’s            |
| PP      | Personal pronoun                              | I, he, it           |
| PPZ     | Possessive pronoun                            | my, his             |
| RB      | Adverb                                        | however, usually    |
| RBR     | Adverb, comparative                           | better              |
| RBS     | Adverb, superlative                           | best                |
| RP      | Particle                                      | give up             |
| SENT    | Sentence-break punctuation                    | . ! ?               |
| SYM     | Symbol                                        | / [ = *             |
| TO      | Infinitive ‘to’                               | to go               |
| UH      | Interjection                                  | uhhuhhuhh           |
| VB      | Verb, base form                               | be                  |
| VBD     | Verb, past tense                              | was, were           |
| VBG     | Verb, gerund/present participle               | being               |
| VBN     | Verb, past participle                         | been                |
| VBP     | Verb, sing. present, non-3d                   | am, are             |
| VBZ     | Verb, 3rd person sing. present                | is                  |
| VH      | Verb have, base form                          | have                |
| VHD     | Verb have, past tense                         | had                 |
| VHG     | Verb have, gerund/present participle          | having              |
| VHN     | Verb have, past participle                    | had                 |
| VHP     | Verb have, sing. present, non-3d              | have                |
| VHZ     | Verb have, 3rd person sing. present           | has                 |
| VV      | Verb, base form                               | take                |
| VVD     | Verb, past tense                              | took                |
| VVG     | Verb, gerund/present participle               | taking              |
| VVN     | Verb, past participle                         | taken               |
| VVP     | Verb, sing. present, non-3d                   | take                |
| VVZ     | Verb, 3rd person sing. present                | takes               |
| WDT     | Wh-determiner                                 | which               |
| WP      | Wh-pronoun                                    | who, what           |
| WP$     | Possessive wh-pronoun                         | whose               |
| WRB     | Wh-abverb                                     | where, when         |

### Main Differences to the Default Penn Tagset
- In TreeTagger:
  - Distinguishes 'be' (VB) and 'have' (VH) from other (non-modal) verbs (VV).
  - For proper nouns, NNP and NNPS have become NP and NPS.
  - SENT for end-of-sentence punctuation (other punctuation tags may also differ).
- In TreeTagger tool + Sketch Engine modifications:
  - The word 'to' is tagged IN when used as a preposition and TO when used as an infinitive marker.

### Bibliography

M. Marcus, B. Santorini and M.A. Marcinkiewicz (1993). Building a large annotated corpus of English: The Penn Treebank. In Computational Linguistics, volume 19, number 2, pp. 313–330.
""")

st.markdown("## Removal of Punctuation")
st.subheader("",divider='rainbow')
st.markdown("""
### Removing punctuation helps to:

1. Reduce the dimensionality of the vector space, making computations more efficient.
2. Reduce noise, leading to more accurate similarity measures.
3. Improve tokenization, ensuring accurate word frequency distributions.
4. Ensure consistent data representation in models like Bag of Words.
""")


def remove_punctuation(text):
    return re.sub(r'[^\w\s]', '', text)
# Input text box
sentence = st.text_input("Enter a sentence:", "Dr. Emily Wong, Ph.D., exclaimed, 'OMG! Despite reaching 7.8 billion in 2021, global population metrics, like those in Section 123(i)(3)(A)(ii), remain baffling,' and then she added in a mix of French and English, 'C’est incroyable, no? Check my blog at www.emilywong-science.com or email me at wong@scienceworld.com for the full story on Mycobacterium tuberculosis complex (MTBC) research, which is, you know, the state-of-the-art stuff—I've literally termed it the Occam’s razor of epidemiology.", key="sentence_input")

if st.button('Submit', key='punctuation_submit'):
    with st.spinner("❌🔡❌"):
       time.sleep(3)
    if sentence:
      # Display the original sentence
      st.subheader("Original Sentence")
      st.write(sentence)
    
      # Remove punctuation from the sentence
      cleaned_sentence = remove_punctuation(sentence)
    
      # Display the cleaned sentence
      st.subheader("Cleaned Sentence (Punctuation Removed)")
      st.write(cleaned_sentence)




st.markdown("## Lowercasing")
st.subheader("",divider='rainbow')
st.markdown("""
### Lowercasing helps with:
1. **Uniformity**: Where the words with different cases "Earth" and "earth" are treated as identical, which results in reducing vocabulary size.
2. **Dimensionality Reduction**: Reducing the numbers of unique tokens which is a must to ensure a smaller vector space.
3. **Model Performance**: Simplifies the features for machine learning models, which leads to better training and evaluation.     
""")



def lowercase(text):
    return text.lower()
# Input text box
sentence = st.text_input("Enter a sentence:", "Dr. Emily Wong, Ph.D., exclaimed, 'OMG! Despite reaching 7.8 billion in 2021, global population metrics, like those in Section 123(i)(3)(A)(ii), remain baffling,' and then she added in a mix of French and English, 'C’est incroyable, no? Check my blog at www.emilywong-science.com or email me at wong@scienceworld.com for the full story on Mycobacterium tuberculosis complex (MTBC) research, which is, you know, the state-of-the-art stuff—I've literally termed it the Occam’s razor of epidemiology.", key="sentence_lower_input")

if st.button('Submit', key='lowercase_submit'):
    with st.spinner("🔠🔡🔠"):
       time.sleep(3)
    if sentence:
      # Display the original sentence
      st.subheader("Original Sentence")
      st.write(sentence)
    
      # Remove punctuation from the sentence
      lowercasing_sentence = lowercase(sentence)
    
      # Display the cleaned sentence
      st.subheader("Cleaned Sentence (Lowercasing)")
      st.write(lowercasing_sentence)



st.markdown("## Stemming")
st.subheader("",divider='rainbow')
st.markdown("""
### Stemming helps with:
1. **Uniformity**: Where the words with different cases "Earth" and "earth" are treated as identical, which results in reducing vocabulary size.
2. **Dimensionality Reduction**: Reducing the numbers of unique tokens which is a must to ensure a smaller vector space.
3. **Model Performance**: Simplifies the features for machine learning models, which leads to better training and evaluation.     
""")




def stemming(words):
    stemmer = PorterStemmer()
    return [stemmer.stem(word) for word in words]
# Input text box
sentence = st.text_input("Enter a sentence:", "Dr. Emily Wong, Ph.D., exclaimed, 'OMG! Despite reaching 7.8 billion in 2021, global population metrics, like those in Section 123(i)(3)(A)(ii), remain baffling,' and then she added in a mix of French and English, 'C’est incroyable, no? Check my blog at www.emilywong-science.com or email me at wong@scienceworld.com for the full story on Mycobacterium tuberculosis complex (MTBC) research, which is, you know, the state-of-the-art stuff—I've literally termed it the Occam’s razor of epidemiology.", key="sentence_stem_input")

if st.button('Submit', key='stem_submit'):
    with st.spinner("🌱🔪🌱"):
       time.sleep(3)
    if sentence:
      # Display the original sentence
      st.subheader("Original Sentence")
      st.write(sentence)
    
      # Remove punctuation from the sentence
      stemming_sentence = stemming(sentence)
    
      # Display the cleaned sentence
      st.subheader("Cleaned Sentence (Stemming)")
      st.write(stemming_sentence)



st.markdown("## Lemmatization")
st.subheader("",divider='rainbow')
st.markdown("""
### Lemmatization helps with:
1. **Uniformity**: Where the words with different cases "Earth" and "earth" are treated as identical, which results in reducing vocabulary size.
2. **Dimensionality Reduction**: Reducing the numbers of unique tokens which is a must to ensure a smaller vector space.
3. **Model Performance**: Simplifies the features for machine learning models, which leads to better training and evaluation.     
""")



def lemmatization(words):
    lemmatizer = WordNetLemmatizer()
    return [lemmatizer.lemmatize(word) for word in words]
# Input text box
sentence = st.text_input("Enter a sentence:", "Dr. Emily Wong, Ph.D., exclaimed, 'OMG! Despite reaching 7.8 billion in 2021, global population metrics, like those in Section 123(i)(3)(A)(ii), remain baffling,' and then she added in a mix of French and English, 'C’est incroyable, no? Check my blog at www.emilywong-science.com or email me at wong@scienceworld.com for the full story on Mycobacterium tuberculosis complex (MTBC) research, which is, you know, the state-of-the-art stuff—I've literally termed it the Occam’s razor of epidemiology.", key="sentence_lemm_input")

if st.button('Submit', key='lemmatize_submit'):
    with st.spinner("📖🔄📖"):
       time.sleep(3)
    if sentence:
      # Display the original sentence
      st.subheader("Original Sentence")
      st.write(sentence)
    
      # Remove punctuation from the sentence
      lemmatization_sentence = lemmatization(sentence)
    
      # Display the cleaned sentence
      st.subheader("Cleaned Sentence (Lemmatization)")
      st.write(lemmatization_sentence)