"""
Runs the scripts needed and handles their results to provide the necessary answers.
"""
import tokenizer
import page_rank
import graph_constructor
from nltk.stem import PorterStemmer
from heapq import nlargest

import math
if __name__ == '__main__':
    #1
    #The first part of this script creates a graph of words as per the exercise instructions.
    print('1. Creating the graph.')
    tokenizedFolder = tokenizer.FolderTokenizer('./www/abstracts',pathToStopWords='./stopwords.txt',wordsToKeep=['NN','NNS','NNP','NNPS','JJ'],stemmer=PorterStemmer())
    folderTokenizer = tokenizedFolder.ngrams

    #2
    #The second part of this script calculates the PageRank score for each word
    print('2. Calculating Page Rank.')
    document_scores = {}
    convergence = 20
    for graph in folderTokenizer:
        document_scores[graph] = page_rank.page_rank(folderTokenizer[graph], 0.85, convergence)

    #3
    #The third part of this script retrieves ngrams and calculates their score.
    #Retrieving 1-grams, 2-grams, 3-grams from the original text.
    print('3. Building ngrams and calculating score.')
    document_multingrams = {}
    for i in range(1,4):
        document_multingrams[i] =  tokenizer.nGramTokenizer('./www/abstracts',stemmer=PorterStemmer(),n=i).ngrams

    #Joining the ngrams into one dictionary with their summed score.
    document_ngrams = {}
    for n in document_multingrams:
        for file in document_multingrams[n]:
            ngrams = document_multingrams[n][file]
            for ngram in ngrams:
                words = ngram.split(' ')
                ngram_score = 0
                for word in words:
                    if word in document_scores[file]:
                        ngram_score += document_scores[file][word]
                document_multingrams[n][file][ngram] = ngram_score
            if file not in document_ngrams:
                document_ngrams[file] = {}
            document_ngrams[file] = {**document_ngrams[file],**document_multingrams[n][file]}

    #4
    #The fourth part of this script retrieves the top k phrases with highest
    #scores and calculates the MRR for it. It also retrieves the golden standards.
    print('4. Calculating the MRR for the Page Rank method.')
    golden_standards = tokenizer.lineTokenizer('./www/gold',stemmer=PorterStemmer()).ngrams
    MRR = {}
    for k in range(1,11):
        MRR[k] = 0
        for file in document_ngrams:
            top = nlargest(k, document_ngrams[file], key=document_ngrams[file].get)
            r = 0
            rd = 0
            for t in top:
                r = r + 1
                if t in golden_standards[file]:
                    rd = r
                    break
            if rd:
                MRR[k] = MRR[k] + 1/rd
        MRR[k] = MRR[k] / len(golden_standards)
        print('MRR when looking at the top {} words: {}'.format(k, MRR[k]))
        with open('MRR.txt','a') as text_file:
            text_file.write('{}\n'.format(MRR[k]))

    #5
    #The fifth part of this script retrieves the term frequency and document
    #frequency of each token and calculates the TF-IDF for each one.
    #It then ranks the highest-scoring words and calculates the MRR for this method.
    print('5. Comparing with TF-IDF.')
    idf = tokenizedFolder.vocabulary
    tf = tokenizedFolder.wordCount
    for i in idf:
        idf[i] = math.log(len(tf)/idf[i],2)
    tf_idf = {}
    for file in tf:
        tf_idf[file]={}
        maxf = nlargest(1, tf[file], key=tf[file].get)
        for word in tf[file]:
            tf[file][word] = tf[file][word] / tf[file][maxf[0]]
            tf_idf[file][word] = tf[file][word] * idf[word]

    MRR_tf_idf = {}
    for k in range(1,11):
        MRR_tf_idf[k] = 0
        for file in tf_idf:
            top = nlargest(k, tf_idf[file], key=tf_idf[file].get)
            r = 0
            rd = 0
            for t in top:
                r = r + 1
                if t in golden_standards[file]:
                    rd = r
                    break
            if rd:
                MRR_tf_idf[k] = MRR_tf_idf[k] + 1/rd
        MRR_tf_idf[k] = MRR_tf_idf[k] / len(golden_standards)
        print('MRR for TF-IDF when looking at the top {} words: {}'.format(k, MRR_tf_idf[k]))
        with open('MRR_tf_idf.txt','a') as text_file:
            text_file.write('{}\n'.format(MRR_tf_idf[k]))
