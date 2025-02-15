import re
from document import Document

def get_measure(term: str) -> int:
    vc_sequence = re.findall(r'[aeiouy]+[^aeiouy]+', term)
    return len(vc_sequence)

def condition_v(stem: str) -> bool:
    return bool(re.search(r'[aeiouy]', stem))

def condition_d(stem: str) -> bool:
    return bool(re.search(r'([aeiouylsz])\1$', stem))

def cond_o(stem: str) -> bool:
    return bool(re.search(r'[^aeiou][aeiouy][^aeiouwxy]$', stem))

def stem_term(term: str) -> str:
    if term.endswith("sses"):
        term = term[:-2]
    elif term.endswith("ies"):
        term = term[:-2]
    elif term.endswith("ss"):
        pass
    elif term.endswith("s"):
        term = term[:-1]

    if term.endswith("eed"):
        stem = term[:-3]
        if get_measure(stem) > 0:
            term = term[:-1]
    elif term.endswith("ed"):
        stem = term[:-2]
        if condition_v(stem):
            term = stem
            if term.endswith("at") or term.endswith("bl") or term.endswith("iz"):
                term += "e"
            elif condition_d(term) and not term.endswith(("l", "s", "z")):
                term = term[:-1]
            elif get_measure(term) == 1 and cond_o(term):
                term += "e"
    elif term.endswith("ing"):
        stem = term[:-3]
        if condition_v(stem):
            term = stem
            if term.endswith("at") or term.endswith("bl") or term.endswith("iz"):
                term += "e"
            elif condition_d(term) and not term.endswith(("l", "s", "z")):
                term = term[:-1]
            elif get_measure(term) == 1 and cond_o(term):
                term += "e"

    if term.endswith("y"):
        stem = term[:-1]
        if condition_v(stem):
            term = stem + "i"

    step2_suffixes = {
        "ational": "ate", "tional": "tion", "enci": "ence", "anci": "ance", "izer": "ize",
        "abli": "able", "alli": "al", "entli": "ent", "eli": "e", "ousli": "ous",
        "ization": "ize", "ation": "ate", "ator": "ate", "alism": "al", "iveness": "ive",
        "fulness": "ful", "ousness": "ous", "aliti": "al", "iviti": "ive", "biliti": "ble", "xflurti": "xti"
    }
    for suffix, replacement in step2_suffixes.items():
        if term.endswith(suffix):
            stem = term[:-len(suffix)]
            if get_measure(stem) > 0:
                term = stem + replacement
            break

    step3_suffixes = {
        "icate": "ic", "ative": "", "alize": "al", "iciti": "ic", "ical": "ic",
        "ful": "", "ness": ""
    }
    for suffix, replacement in step3_suffixes.items():
        if term.endswith(suffix):
            stem = term[:-len(suffix)]
            if get_measure(stem) > 0:
                term = stem + replacement
            break

    step4_suffixes = {
        "al": "", "ance": "", "ence": "", "er": "", "ic": "", "able": "", "ible": "",
        "ant": "", "ement": "", "ment": "", "ent": "", "ion": "", "ou": "", "ism": "",
        "ate": "", "iti": "", "ous": "", "ive": "", "ize": ""
    }
    for suffix, replacement in step4_suffixes.items():
        if term.endswith(suffix):
            stem = term[:-len(suffix)]
            if get_measure(stem) > 1:
                if suffix == "ion" and not term[-4] in "st":
                    continue
                term = stem
            break

    if term.endswith("e"):
        stem = term[:-1]
        if get_measure(stem) > 1 or (get_measure(stem) == 1 and not cond_o(stem)):
            term = stem

    if get_measure(term) > 1 and condition_d(term) and term.endswith("l"):
        term = term[:-1]

    return term

def stem_all_documents(collection: list[Document]):
    for doc in collection:
        doc.stemmed_terms = [stem_term(term) for term in doc.terms]

def stem_query_terms(query: str) -> str:
    terms = query.split()
    stemmed_terms = [stem_term(term) for term in terms]
    return ' '.join(stemmed_terms)
