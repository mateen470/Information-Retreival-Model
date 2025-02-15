# --------------------------------------------------------------------------------
# Information Retrieval SS2024 - Practical Assignment Template
# --------------------------------------------------------------------------------
# This Python template is provided as a starting point for your assignments PR02-04.
# It serves as a base for a very rudimentary text-based information retrieval system.
#
# Please keep all instructions from the task description in mind.
# Especially, avoid changing the base structure, function or class names or the
# underlying program logic. This is necessary to run automated tests on your code.
#
# Instructions:
# 1. Read through the whole template to understand the expected workflow and outputs.
# 2. Implement the required functions and classes, filling in your code where indicated.
# 3. Test your code to ensure functionality and correct handling of edge cases.
#
# Good luck!


import json
import os
import re
import time
import numpy as np
import math

import cleanup
import extraction
import models
import porter
from document import Document

# Important paths:
RAW_DATA_PATH = 'raw_data'
DATA_PATH = 'data'
COLLECTION_PATH = os.path.join(DATA_PATH, 'my_collection.json')
STOPWORD_FILE_PATH = os.path.join(DATA_PATH, 'stopwords.json')
GROUND_TRUTH_PATH = os.path.join(RAW_DATA_PATH, 'ground_truth.txt')


# Menu choices:
(CHOICE_LIST, CHOICE_SEARCH, CHOICE_EXTRACT, CHOICE_UPDATE_STOP_WORDS, CHOICE_SET_MODEL, CHOICE_SHOW_DOCUMENT,
 CHOICE_EXIT) = 1, 2, 3, 4, 5, 6, 9
MODEL_BOOL_LIN, MODEL_BOOL_INV, MODEL_BOOL_SIG, MODEL_FUZZY, MODEL_VECTOR = 1, 2, 3, 4, 5
SW_METHOD_LIST, SW_METHOD_CROUCH = 1, 2


class InformationRetrievalSystem(object):
    def __init__(self):
        if not os.path.isdir(DATA_PATH):
            os.makedirs(DATA_PATH)

        # Collection of documents, initially empty.
        try:
            self.collection = extraction.load_collection_from_json(
                COLLECTION_PATH)
        except FileNotFoundError:
            print('No previous collection was found. Creating empty one.')
            self.collection = []

        # Stopword list, initially empty.
        try:
            with open(STOPWORD_FILE_PATH, 'r') as f:
                self.stop_word_list = json.load(f)
        except FileNotFoundError:
            print('No stopword list was found.')
            self.stop_word_list = []

        self.model = None  # Saves the current IR model in use.
        # Controls how many results should be shown for a query.
        self.output_k = 5

    def main_menu(self):
        """
        Provides the main loop of the CLI menu that the user interacts with.
        """
        while True:
            print(f'Current retrieval model: {self.model}')
            print(f'Current collection: {len(self.collection)} documents')
            print()
            print('Please choose an option:')
            print(f'{CHOICE_LIST} - List documents')
            print(f'{CHOICE_SEARCH} - Search for term')
            print(f'{CHOICE_EXTRACT} - Build collection')
            print(f'{CHOICE_UPDATE_STOP_WORDS} - Rebuild stopword list')
            print(f'{CHOICE_SET_MODEL} - Set model')
            print(f'{CHOICE_SHOW_DOCUMENT} - Show a specific document')
            print(f'{CHOICE_EXIT} - Exit')
            action_choice = int(input('Enter choice: '))

            if action_choice == CHOICE_LIST:
                # List documents in CLI.
                if self.collection:
                    for document in self.collection:
                        print(document)
                else:
                    print('No documents.')
                print()

            elif action_choice == CHOICE_SEARCH:
                # Read a query string from the CLI and search for it.

                # Determine desired search parameters:
                SEARCH_NORMAL, SEARCH_SW, SEARCH_STEM, SEARCH_SW_STEM = 1, 2, 3, 4
                print('Search options:')
                print(f'{SEARCH_NORMAL} - Standard search (default)')
                print(f'{SEARCH_SW} - Search documents with removed stopwords')
                print(f'{SEARCH_STEM} - Search documents with stemmed terms')
                print(
                    f'{SEARCH_SW_STEM} - Search documents with removed stopwords AND stemmed terms')
                search_mode = int(input('Enter choice: '))
                stop_word_filtering = (search_mode == SEARCH_SW) or (
                    search_mode == SEARCH_SW_STEM)
                stemming = (search_mode == SEARCH_STEM) or (
                    search_mode == SEARCH_SW_STEM)

                # Actual query processing begins here:
                query = input('Query: ')

                # TO CALCULATE QUERY PROCESS TIME
                start_time = time.time()

                if stemming:
                    query = porter.stem_query_terms(query)

                if isinstance(self.model, models.InvertedListBooleanModel):
                    results = self.inverted_list_search(
                        query, stemming, stop_word_filtering)
                elif isinstance(self.model, models.VectorSpaceModel):
                    results = self.buckley_lewit_search(
                        query, stemming, stop_word_filtering)
                elif isinstance(self.model, models.SignatureBasedBooleanModel):
                    results = self.signature_search(
                        query, stemming, stop_word_filtering)
                else:
                    results = self.basic_query_search(
                        query, stemming, stop_word_filtering)

                # Output of results:
                for (score, document) in results:
                    print(f'{score}: {document}')

                # TO CALCULATE QUERY PROCESS TIME
                end_time = time.time()

                # Output of quality metrics:
                print()
                print(f'precision: {self.calculate_precision(query,results)}')
                print(f'recall: {self.calculate_recall(query,results)}')
                print(
                    f'Query Process Time: {round(((end_time-start_time)*1000),2)} ms')

            elif action_choice == CHOICE_EXTRACT:
                # Extract document collection from text file.

                raw_collection_file = os.path.join(
                    RAW_DATA_PATH, 'aesopa10.txt')
                self.collection = extraction.extract_collection(
                    raw_collection_file)
                assert isinstance(self.collection, list)
                assert all(isinstance(d, Document) for d in self.collection)

                if input('Should stopwords be filtered? [y/N]: ') == 'y':
                    cleanup.filter_collection(self.collection)

                if input('Should stemming be performed? [y/N]: ') == 'y':
                    porter.stem_all_documents(self.collection)

                extraction.save_collection_as_json(
                    self.collection, COLLECTION_PATH)
                print('Done.\n')

            elif action_choice == CHOICE_UPDATE_STOP_WORDS:
                # Rebuild the stop word list, using one out of two methods.

                print('Available options:')
                print(f'{SW_METHOD_LIST} - Load stopword list from file')
                print(
                    f"{SW_METHOD_CROUCH} - Generate stopword list using Crouch's method")

                method_choice = int(input('Enter choice: '))
                if method_choice in (SW_METHOD_LIST, SW_METHOD_CROUCH):
                    # Load stop words using the desired method:
                    if method_choice == SW_METHOD_LIST:
                        self.stop_word_list = cleanup.load_stop_word_list(
                            os.path.join(RAW_DATA_PATH, 'englishST.txt'))
                        print('Done.\n')
                    elif method_choice == SW_METHOD_CROUCH:
                        self.stop_word_list = cleanup.create_stop_word_list_by_frequency(
                            self.collection)
                        print('Done.\n')

                    # Save new stopword list into file:
                    with open(STOPWORD_FILE_PATH, 'w') as f:
                        json.dump(self.stop_word_list, f)
                else:
                    print('Invalid choice.')

            elif action_choice == CHOICE_SET_MODEL:
                # Choose and set the retrieval model to use for searches.

                print()
                print('Available models:')
                print(f'{MODEL_BOOL_LIN} - Boolean model with linear search')
                print(f'{MODEL_BOOL_INV} - Boolean model with inverted lists')
                print(
                    f'{MODEL_BOOL_SIG} - Boolean model with signature-based search')
                print(f'{MODEL_FUZZY} - Fuzzy set model')
                print(f'{MODEL_VECTOR} - Vector space model')
                model_choice = int(input('Enter choice: '))
                if model_choice == MODEL_BOOL_LIN:
                    self.model = models.LinearBooleanModel(self.collection)
                elif model_choice == MODEL_BOOL_INV:
                    self.model = models.InvertedListBooleanModel(
                        self.collection)
                elif model_choice == MODEL_BOOL_SIG:
                    self.model = models.SignatureBasedBooleanModel(
                        self.collection)
                elif model_choice == MODEL_FUZZY:
                    self.model = models.FuzzySetModel()
                elif model_choice == MODEL_VECTOR:
                    self.model = models.VectorSpaceModel(self.collection)
                else:
                    print('Invalid choice.')

            elif action_choice == CHOICE_SHOW_DOCUMENT:
                target_id = int(input('ID of the desired document:'))
                found = False
                for document in self.collection:
                    if document.document_id == target_id:
                        print(document.title)
                        print('-' * len(document.title))
                        print(document.raw_text)
                        found = True

                if not found:
                    print(f'Document #{target_id} not found!')

            elif action_choice == CHOICE_EXIT:
                break
            else:
                print('Invalid choice.')

            print()
            input('Press ENTER to continue...')
            print()

    def evaluate_expression(self, tokens, get_term_docs, eval_not, eval_and, eval_or):
        ops = []
        values = []
        precedence = {'-': 3, '&': 2, '|': 1, '(': 0, ')': 0}

        def apply_operator():
            operator = ops.pop()
            if operator == '&':
                right = values.pop()
                left = values.pop()
                values.append(eval_and(left, right))
            elif operator == '|':
                right = values.pop()
                left = values.pop()
                values.append(eval_or(left, right))
            elif operator == '-':
                value = values.pop()
                values.append(eval_not(value))

        for token in tokens:
            if token.isalnum():
                values.append(get_term_docs(token))
            elif token == '(':
                ops.append(token)
            elif token == ')':
                while ops and ops[-1] != '(':
                    apply_operator()
                ops.pop()
            else:
                while (ops and precedence[ops[-1]] >= precedence[token]):
                    apply_operator()
                ops.append(token)

        while ops:
            apply_operator()

        return values[0] if values else set()

    def basic_query_search(self, query: str, stemming: bool, stop_word_filtering: bool) -> list:
        tokens = re.findall(r'\(|\)|\w+|&|\||-', query)

        def get_term_docs(term):
            return {doc.document_id for doc in self.collection if term in doc.terms}

        def eval_not(term_set):
            all_docs = set(doc.document_id for doc in self.collection)
            return all_docs - term_set

        def eval_and(set1, set2):
            return set1 & set2

        def eval_or(set1, set2):
            return set1 | set2

        matching_docs_ids = self.evaluate_expression(
            tokens, get_term_docs, eval_not, eval_and, eval_or)

        results = [(1, next(doc for doc in self.collection if doc.document_id == doc_id))
                   for doc_id in sorted(matching_docs_ids)]
        return results[:self.output_k]

    def inverted_list_search(self, query: str, stemming: bool, stop_word_filtering: bool) -> list:
        if not hasattr(self, 'inverted_index'):
            self.inverted_index = {}
            for document in self.collection:
                terms = document.terms
                if stop_word_filtering:
                    terms = document.filtered_terms
                if stemming:
                    terms = document.stemmed_terms

                for term in terms:
                    if term not in self.inverted_index:
                        self.inverted_index[term] = set()
                    self.inverted_index[term].add(document.document_id)

        def eval_term(term):
            return self.inverted_index.get(term, set())

        def eval_not(term_set):
            all_docs = set(doc.document_id for doc in self.collection)
            return all_docs - term_set

        def eval_and(set1, set2):
            return set1 & set2

        def eval_or(set1, set2):
            return set1 | set2

        tokens = re.findall(r'\(|\)|\w+|&|\||-', query)
        matching_docs_ids = self.evaluate_expression(
            tokens, eval_term, eval_not, eval_and, eval_or)

        ranked_results = [(1, next(doc for doc in self.collection if doc.document_id == doc_id))
                          for doc_id in sorted(matching_docs_ids)]
        return ranked_results[:self.output_k]

    def buckley_lewit_search(self, query: str, stemming: bool, stop_word_filtering: bool) -> list:
        query_terms = query.lower().split()
        N = len(self.collection)

        query_vector = np.zeros(len(self.model.inverted_index))
        for term in query_terms:
            if term in self.model.inverted_index:
                tf = query_terms.count(term)
                df = len(self.model.inverted_index[term])
                idf = math.log(N / df)
                term_index = list(self.model.inverted_index.keys()).index(term)
                query_vector[term_index] = (
                    0.5 + (0.5 * tf / max(query_terms.count(t) for t in query_terms))) * idf
        norm = np.linalg.norm(query_vector)
        if norm != 0:
            query_vector = query_vector / norm

        scores = np.zeros(N)
        for idx, doc_vector in self.model.document_vectors.items():
            scores[idx] = np.dot(doc_vector, query_vector)

        top_k_indices = scores.argsort()[::-1][:self.output_k]

        results = [(scores[idx], self.collection[idx])
                   for idx in top_k_indices if scores[idx] > 0]

        return results

    def signature_search(self, query: str, stemming: bool, stop_word_filtering: bool) -> list:
        def search_documents(model, query):
            results = set()

            if "&" in query:
                terms = query.split('&')
                query_signature = model.query_to_representation(terms[0])
                query_signature |= model.query_to_representation(terms[1])

                term_result = model.match(query_signature)
                results = results.union(term_result)

            elif "|" in query:
                terms = query.split('|')

                query_signature = model.query_to_representation(terms[0])
                term_1_result = model.match(query_signature)
                results = results.union(term_1_result)

                query_signature = model.query_to_representation(terms[1])
                term_2_result = model.match(query_signature)
                results = results.union(term_2_result)
            else:
                query_signature = model.query_to_representation(query)
                term_result = model.match(query_signature)
                results = results.union(term_result)

            return results

        def remove_false_drop(model, query, matched_documents_ids):
            results = set()

            if "&" in query:
                terms = query.split('&')
                query_seperation_1 = [
                    doc.document_id for doc in self.collection if doc.document_id in matched_documents_ids and terms[0] in doc.terms
                ]
                query_seperation_2 = [
                    doc.document_id for doc in self.collection if doc.document_id in matched_documents_ids and terms[1] in doc.terms
                ]

                results = set(query_seperation_1).intersection(
                    set(query_seperation_2))

            elif "|" in query:
                terms = query.split('|')
                query_seperation_1 = [
                    doc.document_id for doc in self.collection if doc.document_id in matched_documents_ids and terms[0] in doc.terms
                ]
                query_seperation_2 = [
                    doc.document_id for doc in self.collection if doc.document_id in matched_documents_ids and terms[1] in doc.terms
                ]
                results = set(query_seperation_1).union(
                    set(query_seperation_2))

            else:
                results = set([
                    doc.document_id for doc in self.collection if doc.document_id in matched_documents_ids and query in doc.terms
                ])

            return results

        matched_documents_ids = search_documents(self.model, query)

        result_list = remove_false_drop(
            self.model, query, matched_documents_ids)

        matched_documents = [
            (1, doc) for doc in self.collection if doc.document_id in result_list
        ]

        matched_documents.sort(reverse=True, key=lambda x: x[0])
        return matched_documents[:self.output_k]

    def load_ground_truth(self, ground_truth_path):
        ground_truth = {}
        with open(ground_truth_path, 'r') as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith('#'):
                    continue
                if ' - ' not in line:
                    print(f'Invalid line in ground_truth.txt: {line}')
                    continue
                term, doc_ids = line.split(' - ', 1)
                try:
                    ground_truth[term] = set(map(int, doc_ids.split(', ')))
                except ValueError as e:
                    print(f'Error processing line: {line} -> {e}')
                    continue
        return ground_truth

    def calculate_precision(self, query: str, results: list[tuple]) -> float:
        ground_truth = self.load_ground_truth(GROUND_TRUTH_PATH)
        query_terms = re.findall(r'\w+|&|\||-', query.lower())
        relevant_docs = set()
        not_operator = '-' in query_terms

        terms = []
        for term in query_terms:
            if term not in ('&', '|', '-'):
                terms.append(term)

        if not terms:
            return -1

        for term in terms:
            if term in ground_truth:
                relevant_docs.update(ground_truth[term])

        if not_operator:
            total_docs = set(doc.document_id for doc in self.collection)
            relevant_docs = total_docs - relevant_docs

        if not relevant_docs:
            return -1

        retrieved_docs = [doc.document_id for score, doc in results]
        retrieved_set = set(retrieved_docs)
        true_positives = retrieved_set.intersection(relevant_docs)

        precision = len(true_positives) / \
            len(retrieved_set) if retrieved_set else -1
        return precision

    def calculate_recall(self, query: str, results: list[tuple]) -> float:

        ground_truth = self.load_ground_truth(GROUND_TRUTH_PATH)
        query_terms = query.lower().split()
        relevant_docs = set()

        for term in query_terms:
            if term in ground_truth:
                relevant_docs.update(ground_truth[term])

        if not relevant_docs:
            return -1

        retrieved_docs = [doc.document_id for score, doc in results]
        retrieved_set = set(retrieved_docs)
        true_positives = retrieved_set.intersection(relevant_docs)

        recall = len(true_positives) / \
            len(relevant_docs) if relevant_docs else -1
        return recall


if __name__ == '__main__':
    irs = InformationRetrievalSystem()
    irs.main_menu()
    exit(0)
