import json
import re
from document import Document


def read_and_skip_header(file, skip_count):
    # REMOVE LINES FROM FILE
    for _ in range(skip_count):
        next(file)
    return file.read()

def create_document(document_id: int, title: str, raw_text: str) -> Document:
    # CREATE NEW DOC OBJECT AND INITIALZES IT WITH PROVIDED PARAMS
    document = Document()
    document.document_id = document_id
    document.title = title
    document.raw_text = raw_text
    document.terms = raw_text.lower().split()
    document.terms = [term for term in document.terms if len(term) > 0]

    return document


def extract_collection(source_file_path: str) -> list[Document]:
    # INITIALIZE VARIABLES TO STORE DOCUMENT DATA
    catalog = []
    id_counter = 1
    active_title = ""
    content_accumulator = []
    empty_line_count = 0
    THRESHOLD_BLANK_LINES = 3

    # READ FILE AND PROCESS ITS CONTENT
    with open(source_file_path, 'r', encoding='utf-8') as file:
        text = read_and_skip_header(file, 304)
        for line in text.splitlines():
            line = line.strip()
            if not line:
                empty_line_count += 1
                continue

            # CHECK IF NEW DOCUMENT SHOULD START
            if empty_line_count >= THRESHOLD_BLANK_LINES:
                if content_accumulator:
                    complete_text = ' '.join(content_accumulator)
                    new_document = create_document(
                        id_counter, active_title, complete_text)
                    catalog.append(new_document)
                    id_counter += 1
                    content_accumulator = []

                active_title = line
                empty_line_count = 0
            else:
                content_accumulator.append(line)
                empty_line_count = 0

        # ENSURE LAST DOCUMENT IS ADDED
        if content_accumulator:
            complete_text = ' '.join(content_accumulator)
            new_document = create_document(
                id_counter, active_title, complete_text)
            catalog.append(new_document)

    return catalog


def save_collection_as_json(collection: list[Document], file_path: str) -> None:
    """
    Saves the collection to a JSON file.
    :param collection: The collection to store (= a list of Document objects)
    :param file_path: Path of the JSON file
    """

    serializable_collection = []
    for document in collection:
        serializable_collection += [{
            'document_id': document.document_id,
            'title': document.title,
            'raw_text': document.raw_text,
            'terms': document.terms,
            'filtered_terms': document.filtered_terms,
            'stemmed_terms': document.stemmed_terms
        }]

    with open(file_path, "w") as json_file:
        json.dump(serializable_collection, json_file)


def load_collection_from_json(file_path: str) -> list[Document]:
    """
    Loads the collection from a JSON file.
    :param file_path: Path of the JSON file
    :return: list of Document objects
    """
    try:
        with open(file_path, "r") as json_file:
            json_collection = json.load(json_file)

        collection = []
        for doc_dict in json_collection:
            document = Document()
            document.document_id = doc_dict.get('document_id')
            document.title = doc_dict.get('title')
            document.raw_text = doc_dict.get('raw_text')
            document.terms = doc_dict.get('terms')
            document.filtered_terms = doc_dict.get('filtered_terms')
            document.stemmed_terms = doc_dict.get('stemmed_terms')
            collection += [document]

        return collection
    except FileNotFoundError:
        print('No collection was found. Creating empty one.')
        return []
