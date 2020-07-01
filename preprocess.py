"""
The purpose of this file is to perform some standard text preprocessing 
so feature extraction and other steps can be performed before running it
through a predictive model.
"""
import os
import re
from nltk import word_tokenize, pos_tag

TEXT_REPO = "data/"

#* open a directory named 'data', for each file in 'data' open it up
for filename in os.listdir(TEXT_REPO):
    with open(f"{TEXT_REPO}/{filename}", 'r') as f:
        text = f.read()
        START_POSITION = re.search(r"\*{3}.+\*{3}", text).end()
        true_text = text[START_POSITION:]
        tokens = word_tokenize(true_text) #* tokenize the text
        print(pos_tag(tokens[:50])) #* generate POS tags