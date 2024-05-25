# Install necessary libraries
pip install torch torchvision torchaudio

# Download Spacy models
python -m spacy download en_core_web_md
python -m spacy download en_core_web_lg

# Download NLTK data
python -m nltk.downloader punkt
python -m nltk.downloader stopwords