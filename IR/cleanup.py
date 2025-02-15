import json
import re
from document import Document
import os
import collections

def remove_symbols(text_string: str) -> str:
    #REMOVES ALL PUNCTUATION MARKS AND SIMILAR SYMBOLS FROM A GIVEN STRING, INCLUDING 'S ENDINGS.
    text_string = re.sub(r"[^\w\s]", "", text_string)  
    return re.sub(r"'s\b", "", text_string)  

def is_stop_word(term: str, stop_word_list: list[str]) -> bool:
    #CHECKS IF A GIVEN TERM IS A STOP WORD BY COMPARING IT AGAINST THE STOP WORD LIST.
    return term.lower() in stop_word_list

def remove_stop_words_from_term_list(terms: list[str]) -> list[str]:
    #PROCESSES A LIST OF TERMS BY ELIMINATING ANY TERMS IDENTIFIED AS STOP WORDS. REMOVES STOP WORDS FROM THE LIST OF TERMS PROVIDED. 
    path_to_stopwords = os.path.join('data', 'stopwords.json')
    
    with open(path_to_stopwords, 'r') as file:
        stop_words = json.load(file)

    non_stop_words = []
    
    for term in terms:
        cleaned_term = remove_symbols(term) 
        if not is_stop_word(cleaned_term, stop_words):
            non_stop_words.append(cleaned_term)
    
    return non_stop_words

def filter_collection(collection: list[Document]):
   #APPLIES STOP WORD FILTERING TO EACH DOCUMENT IN THE PROVIDED COLLECTION.
    for document in collection:
        document.filtered_terms = remove_stop_words_from_term_list(document.terms)

def load_stop_word_list(raw_file_path: str) -> list[str]:
   #LOADS A TEXT FILE THAT CONTAINS STOP WORDS AND RETURNS THEM AS A LIST.
    with open(raw_file_path, 'r', encoding='utf-8') as file:
        return [line.strip().lower() for line in file if line.strip()]

def create_stop_word_list_by_frequency(collection: list[Document]) -> list[str]:
    #GENERATES A STOP WORD LIST BY IDENTIFYING HIGH AND LOW FREQUENCY TERMS WITHIN THE PROVIDED DOCUMENT COLLECTION.
    high_frequency_limit = 50
    low_frequency_limit = 2
    
    all_terms = []
    for document in collection:
        all_terms += [remove_symbols(term) for term in document.terms]
        
    frequency_of_terms = collections.Counter(all_terms)
    
    high_frequency_of_terms = {term for term, tf in frequency_of_terms.items() if tf >= high_frequency_limit}
    low_frequency_of_terms = {term for term, tf in frequency_of_terms.items() if tf < low_frequency_limit}
    
    final_stop_words_list = high_frequency_of_terms.union(low_frequency_of_terms)
    
    return list(final_stop_words_list)

