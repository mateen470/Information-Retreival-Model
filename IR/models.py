
from abc import ABC, abstractmethod
from typing import List, Dict
from document import Document
import numpy as np
import math


class RetrievalModel(ABC):
    @abstractmethod
    def document_to_representation(self, document: Document, stopword_filtering=False, stemming=False):
        return

    @abstractmethod
    def query_to_representation(self, query: str):
        raise NotImplementedError()

    @abstractmethod
    def match(self, document_representation, query_representation) -> float:
        raise NotImplementedError()


class LinearBooleanModel(RetrievalModel):
    def __init__(self, collection: List[Document]):
        self.collection = collection

    def __str__(self):
        return 'Boolean Model (Linear)'

    def document_to_representation(self, document: Document, stopword_filtering=False, stemming=False):
        # CONVERT A DOCUMENT INTO A SEARCHABLE REPRESENTATION BASED ON THE CONFIGURED FILTERS.
        if stopword_filtering:
            return document.filtered_terms
        else:
            return document.terms

    def query_to_representation(self, query: str):
        # CONVERT QUERY STRING TO LOWERCASE FOR NORMALIZATION.
        return query.lower()

    def match(self, document_representation, query_representation) -> float:
        # DETERMINE IF THE QUERY IS PRESENT IN THE DOCUMENT'S REPRESENTATION.
        return 1.0 if query_representation in document_representation else 0.0

    def search(self, term: str) -> List[Document]:
        # SEARCH THROUGH THE COLLECTION FOR DOCUMENTS MATCHING THE QUERY TERM.
        term = self.query_to_representation(term)
        matching_documents = []
        for doc in self.collection:
            document_rep = self.document_to_representation(doc)
            if self.match(document_rep, term):
                matching_documents.append(doc)
        return matching_documents


class InvertedListBooleanModel(RetrievalModel):
    def __init__(self, collection):
        # INITIALIZE THE MODEL WITH A GIVEN DOCUMENT COLLECTION
        self.collection = collection
        self.inverted_index = self.build_inverted_index()

    def build_inverted_index(self):
        # BUILD AN INVERTED INDEX FROM THE DOCUMENT COLLECTION
        inverted_index = {}
        for doc in self.collection:
            doc_id = doc.document_id
            for term in doc.terms:
                if term not in inverted_index:
                    inverted_index[term] = set()
                inverted_index[term].add(doc_id)
        return inverted_index

    def document_to_representation(self, document: Document, stopword_filtering=False, stemming=False):
        # CONVERT A DOCUMENT INTO A SET OF TERMS, APPLYING STOPWORD FILTERING AND STEMMING AS NEEDED
        terms = document.terms
        if stopword_filtering:
            terms = document.filtered_terms
        if stemming:
            terms = document.stemmed_terms
        return set(terms)

    def query_to_representation(self, query: str):
        # CONVERT A QUERY STRING INTO A SET OF LOWERCASE TERMS
        return set(query.lower().split())

    def match(self, document_representation, query_representation) -> float:
        # CALCULATE THE MATCH SCORE BETWEEN A DOCUMENT REPRESENTATION AND A QUERY REPRESENTATION
        return float(len(document_representation & query_representation) / len(query_representation) if query_representation else 0)

    def search(self, query: str, stopword_filtering=False, stemming=False) -> list:
        # SEARCH FOR DOCUMENTS THAT MATCH THE GIVEN QUERY, WITH OPTIONAL STOPWORD FILTERING AND STEMMING
        query_rep = self.query_to_representation(query)
        results = []
        for term in query_rep:
            if term in self.inverted_index:
                doc_ids = self.inverted_index[term]
                for doc_id in doc_ids:
                    document = next(
                        doc for doc in self.collection if doc.document_id == doc_id)
                    results.append(document)
        return results


class SignatureBasedBooleanModel:
    def __init__(self, collection):
        # INITIALIZE THE MODEL WITH A GIVEN DOCUMENT COLLECTION AND SOME USEABLE VARIBLES AND SIGNATURE OF EACH DOCUMENT IS BUILT
        self.collection = collection
        self.signatures = {}
        self.F = 64
        self.D = 4
        self.p = 157
        self.m = 5
        self.build_signature_index()

    def build_signature_index(self):
        # TERMS ARRAY IS TAKEN FROM EACH DOCUMENT AND THEN D-SIZED BLOCK CHUNKS WITHIN THAT TERMS ARRAY IS SIGNATURIZED AND OR-CONJUGATED TO CREATE BLOCK SIGNATURES
        for doc in self.collection:
            document_signatures = []
            terms = doc.terms

            for i in range(0, len(terms), self.D):
                block_terms = terms[i:i+self.D]
                block_signature = np.zeros(self.F, dtype=int)
                for term in block_terms:
                    term_signature = self.generate_signature(term)
                    block_signature |= term_signature
                document_signatures.append(
                    block_signature)

            self.signatures[doc.document_id] = document_signatures

    def generate_signature(self, word):
        # A TERM IS PASSED, THEN IT IS HASHED AND CONVERTED INTO F-SIZED BINARY SIGNATURE AND 1's ARE PLACED ACCORDINGLY
        signature = np.zeros(self.F, dtype=int)

        for i in range(self.m):
            hash_value = 0
            for char in word:
                hash_value = (hash_value + ord(char)) * (i * self.p)
            hash_value = hash_value % self.F
            signature[hash_value] = 1

        return signature

    def document_to_representation(self, document: Document, stopword_filtering=False, stemming=False):
        # RETURNS THE SIGNATURES AGAINST THE PROVIDED DOCUMENT-ID
        return self.signatures.get(document.document_id, [])

    def query_to_representation(self, query: str):
        # CREATE QUERY SIGNATURE
        return self.generate_signature(query.lower())

    def match(self, query_signature):
        #MATCHING QUERY AND BLOCK SIGNATURE
        doc_result = set()
    
        for did, signatures in self.signatures.items():
            for sign in signatures:
                result = np.all((sign & query_signature) == query_signature)
                if result:
                    doc_result.add(did)
        
        return doc_result


class VectorSpaceModel(RetrievalModel):
    def __init__(self, collection: List[Document]):
        # INITIALIZE VECTOR SPACE MODEL WITH DOCUMENT COLLECTION
        self.collection = collection
        self.inverted_index = self.build_inverted_index()
        self.document_vectors = self.build_document_vectors()

    def __str__(self):
        return 'Vector Space Model'

    def build_inverted_index(self) -> Dict[str, List[int]]:
        # CONSTRUCT AN INVERTED INDEX FOR TERM DOCUMENT FREQUENCY
        inverted_index = {}
        for idx, doc in enumerate(self.collection):
            for term in doc.terms:
                if term not in inverted_index:
                    inverted_index[term] = []
                inverted_index[term].append(idx)
        return inverted_index

    def build_document_vectors(self) -> Dict[int, np.ndarray]:
        # GENERATE VECTOR REPRESENTATIONS FOR ALL DOCUMENTS BASED ON TF-IDF
        doc_vectors = {}
        N = len(self.collection)

        for idx, doc in enumerate(self.collection):
            vector = np.zeros(len(self.inverted_index))
            for term in doc.terms:
                if term in self.inverted_index:
                    tf = doc.terms.count(term)
                    df = len(self.inverted_index[term])
                    idf = math.log(N / df)
                    term_index = list(self.inverted_index.keys()).index(term)
                    vector[term_index] = tf * idf
            norm = np.linalg.norm(vector)
            if norm != 0:
                vector = vector / norm
            doc_vectors[idx] = vector

        return doc_vectors

    def document_to_representation(self, document: Document, stopword_filtering=False, stemming=False):
        # CONVERT DOCUMENT INTO ITS VECTOR REPRESENTATION BASED ON INDEXED TERMS
        doc_id = self.collection.index(document)
        return self.document_vectors[doc_id]

    def query_to_representation(self, query: str):
        # CONVERT QUERY INTO VECTOR FORM USING TF-IDF WEIGHTING
        query_terms = query.lower().split()
        query_vector = np.zeros(len(self.inverted_index))
        N = len(self.collection)
        for term in query_terms:
            if term in self.inverted_index:
                tf = query_terms.count(term)
                df = len(self.inverted_index[term])
                idf = math.log(N / df)
                term_index = list(self.inverted_index.keys()).index(term)
                query_vector[term_index] = (
                    0.5 + (0.5 * tf / max(query_terms.count(t) for t in query_terms))) * idf
        norm = np.linalg.norm(query_vector)
        if norm != 0:
            query_vector = query_vector / norm
        return query_vector

    def match(self, document_representation, query_representation) -> float:
        # CALCULATE COSINE SIMILARITY BETWEEN DOCUMENT VECTOR AND QUERY VECTOR
        return np.dot(document_representation, query_representation)

    def search(self, query: str) -> List[Document]:
        # PERFORM QUERY SEARCH ACROSS DOCUMENTS USING VECTOR REPRESENTATIONS
        query_vector = self.query_to_representation(query)
        scores = []
        for idx, doc_vector in self.document_vectors.items():
            score = self.match(doc_vector, query_vector)
            scores.append((score, self.collection[idx]))
        scores.sort(reverse=True, key=lambda x: x[0])
        return [doc for score, doc in scores]


class FuzzySetModel(RetrievalModel):
    # TODO: Implement all abstract methods. (PR04)
    def __init__(self):
        # TODO: Remove this line and implement the function.
        raise NotImplementedError()

    def __str__(self):
        return 'Fuzzy Set Model'
