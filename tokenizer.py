# -*- coding: utf-8 -*-
"""
This script tokenizes the input data and answers the excercise's questions
This exercise solution is presented by Rogelio Garcia.
"""
import os
import re
from nltk.tokenize import WhitespaceTokenizer
from nltk.stem import PorterStemmer
from nltk.corpus import stopwords
import graph_constructor
class FolderTokenizer:
    """
    This class has all the functions necessary to tokenize a folder of files.
    Inputs:
        pathToFiles - string containing the route to the folder that contains the files to tokenize
        pathToStopWords - if there is a stopwords catalogue to be used, it will eliminate stop words from the tokenized output
    This class builds a vocabulary, calculates total word count and word count per document
    Methods used:
        *Total word count checks if each encountered token is new and either adds it
        or sums one to the existing value, regardless of the document it was present in
        *Number of words sums one for each token encountered, disregarding if its a new token or not
        *Vocabulary builds an inversed index matrix with a hash table in the form
        of a Python Dictionary. Each entry in the table has another hash table with
        each document where the token was present in and the frequency inside that document
    """
    def __init__(self, pathToFiles, pathToStopWords=None, eliminateStopWords=False, stemmer=None, wordsToKeep=None, n=1):
        """
        Initializes the class with parameters
            pathToFiles - string containing the route to the folder that contains the files to tokenize
            pathToStopWords - if there is a stopwords catalogue to be used, it will eliminate stop words from the tokenized output
            eliminateStopWords - Boolean, if True removes stopwords, if False, ignores stopwords and adds them to the vocabulary.
            stemmer - The stemmer to be used. It has to have a "stem" method that accepts a word (String) and outputs a string.
        The stop words are stored in a list called stopWords, and it expects the stop words catalogue in the format:
            word1
            word2
            .
            .
            .
            wordn
        with each word on a single line.

        This function initializes the following class variables:
            pathToFiles: same as parameter.
            vocabulary: dictionary (hash table) containing the vocabulary in inverted index notation.
            numberOfWords: keeps track of the total number of words in the collection.
            wordCount: dictionary containing the vocabulary as a total sum of frequencies in the collection.
            stopWords: catalogue of stopwords. May be None.
            stemmer: same as parameter.
            eliminateStopWords: same as parameter.
        """
        self.pathToFiles = pathToFiles
        self.vocabulary = {}
        self.numberOfWords = 0
        self.wordCount = {}
        self.stopWords = []
        self.stemmer = stemmer
        self.eliminateStopWords = eliminateStopWords
        self.wordsToKeep = wordsToKeep
        self.n=n
        self.ngrams = {}
        if pathToStopWords:
            with open('./{}'.format(pathToStopWords)) as stopWordsFile:
                self.stopWords = [line.rstrip('\n') for line in stopWordsFile]
        self.iterateThroughFiles()

    def iterateThroughFiles(self):
        """
        This function iterates through every file in the folder and calls the
        tokenizer function for each file in the folder.
        """
        for filename in os.listdir(self.pathToFiles):
            with open('./{}/{}'.format(self.pathToFiles, filename)) as currentFile:
                self.tokenizer(currentFile, filename)

    def tokenizer(self, currentFile, filename):
        """
        This function receives a file and adds each token identified to the vocabulary.
        It removes punctuation, various special characters and numbers, and it
        separates tokens on spaces.
        """
        self.ngrams[filename] = graph_constructor.Undirected_graph()
        lastTokens = []
        for line in currentFile:
            tokens = WhitespaceTokenizer().tokenize(line)
            for token in tokens:
                token = token.split("_")
                if token[1] not in self.wordsToKeep:
                    lastTokens = []
                    continue
                token = token[0].lower()
                if self.checkStopWord(token):
                    lastTokens = []
                    continue
                if self.stemmer:
                    token = self.stemmer.stem(token)
                if not token:
                    continue
                if token not in self.vocabulary:
                    self.vocabulary[token] = 1
                    self.ngrams[filename].add_node(token)
                else:
                    self.vocabulary[token] += 1
                if lastTokens:
                    for lastToken in lastTokens:
                        currentEdge = self.ngrams[filename].get_edge(lastToken, token)
                        if currentEdge == -1:
                            self.ngrams[filename].add_edge(lastToken, token, 1)
                        else:
                            self.ngrams[filename].add_edge(lastToken, token, currentEdge+1)
                lastTokens.append(token)
                if len(lastTokens) > self.n:
                    lastTokens = lastTokens[1:]

    def checkStopWord(self,word):
        """
        If there is a stop word catalogue, it checks if a given word is a stop word in the catalogue.
        If not, it checks if a give word is a stop word in the NLTK catalogue.
        It receives the word as parameters and outputs True or False.
        """
        if not self.stopWords:
            return False
        else:
            if word in self.stopWords:
                return True
            else:
                return False

    def sparseVectors(self):
        """
        Returns the vector in the sparse TF-IDF notation.
        Returns a dictionary of dictionaries.
        """
        words = list(self.vocabulary.keys())
        sparseVectors = {}
        currentWord = 0
        for word in words:
            currentWord += 1
            files = self.vocabulary[word]
            for file in files:
                if file not in sparseVectors:
                    sparseVectors[file] = {currentWord:self.vocabulary[word][file]}
                else:
                    sparseVectors[file][currentWord] = self.vocabulary[word][file]
        return sparseVectors

    def listOfWords(self):
        """
        Returns a list of the words in the vocabulary.
        The index for each word matches the index used in the sparse vector.
        """
        return list(self.vocabulary.keys())

class nGramTokenizer:
    def __init__(self, pathToFiles,stemmer=None,n=1):
        self.pathToFiles = pathToFiles
        self.stemmer = stemmer
        self.n=n
        self.ngrams = {}
        self.iterateThroughFiles()

    def iterateThroughFiles(self):
        for filename in os.listdir(self.pathToFiles):
            with open('./{}/{}'.format(self.pathToFiles, filename)) as currentFile:
                self.tokenizer(currentFile, filename)

    def tokenizer(self, currentFile, filename):
        self.ngrams[filename] = {}
        lastTokens = []
        for line in currentFile:
            tokens = WhitespaceTokenizer().tokenize(line)
            for token in tokens:
                token = token.split("_")
                token = token[0].lower()
                if self.stemmer:
                    token = self.stemmer.stem(token)
                if not token:
                    continue
                lastTokens.append(token)
                if len(lastTokens) > self.n:
                    lastTokens = lastTokens[1:]
                newNGram = ''
                if len(lastTokens) == self.n:
                    for currentToken in lastTokens:
                        newNGram = '{} {}'.format(newNGram,currentToken)
                    newNGram = newNGram.strip()
                if newNGram:
                    self.ngrams[filename][newNGram] = 0

class lineTokenizer:
    def __init__(self, pathToFiles,stemmer=None):
        self.pathToFiles = pathToFiles
        self.stemmer = stemmer
        self.ngrams = {}
        self.iterateThroughFiles()

    def iterateThroughFiles(self):
        for filename in os.listdir(self.pathToFiles):
            with open('./{}/{}'.format(self.pathToFiles, filename)) as currentFile:
                self.tokenizer(currentFile, filename)

    def tokenizer(self, currentFile, filename):
        lines = [line.rstrip('\n') for line in currentFile]
        self.ngrams[filename] = {}
        lastTokens = []
        for line in lines:
            tokens = WhitespaceTokenizer().tokenize(line)
            lastTokens = []
            for token in tokens:
                token = token.split("_")
                token = token[0].lower()
                if self.stemmer:
                    token = self.stemmer.stem(token)
                if not token:
                    continue
                lastTokens.append(token)
            newNGram = ''
            for currentToken in lastTokens:
                newNGram = '{} {}'.format(newNGram,currentToken)
            newNGram = newNGram.strip()
            if newNGram:
                self.ngrams[filename][newNGram] = 0

if __name__ == '__main__':
    folderTokenizer = FolderTokenizer('./wwwSmall/abstracts',pathToStopWords='./stopwords.txt',wordsToKeep=['NN','NNS','NNP','NNPS','JJ'],stemmer=PorterStemmer())
