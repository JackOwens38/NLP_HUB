#!/bin/bash
python -m spacy download en_core_web_md
python -m spacy download en_core_web_lg
python -m nltk.downloader punkt
python -m nltk.downloader stopwords