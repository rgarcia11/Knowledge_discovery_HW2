"""
Runs the scripts necessary to answer the question
"""
import tokenizer
import page_rank
import graph_constructor
from nltk.stem import PorterStemmer
from heapq import nlargest
if __name__ == '__main__':
    #1
    folderTokenizer = tokenizer.FolderTokenizer('./wwwSmall/abstracts',pathToStopWords='./stopwords.txt',wordsToKeep=['NN','NNS','NNP','NNPS','JJ'],stemmer=PorterStemmer()).ngrams
    document_scores = {}
    document_multingrams = {}
    convergence = 20

    #2
    for graph in folderTokenizer:
        document_scores[graph] = page_rank.page_rank(folderTokenizer[graph], 0.85, convergence)
        print(document_scores[graph])
    #3
    for i in range(1,4):
        document_multingrams[i] =  tokenizer.nGramTokenizer('./wwwSmall/abstracts',stemmer=PorterStemmer(),n=i).ngrams

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
    print(document_ngrams)

    #4
    golden_standards = tokenizer.lineTokenizer('./wwwSmall/gold',stemmer=PorterStemmer()).ngrams
    print('')
    print(golden_standards)
    print('')
    print(document_ngrams)
    for k in range(1,11):
        MRR = 0
        #print('The top-{} ranked n-grams.'.format(k))
        for file in document_ngrams:
            #print('The top-{} ranked n-grams of file {}.'.format(k,file))
            top = nlargest(k, document_ngrams[file], key=document_ngrams[file].get)
            r = 0
            rd = 0
            for t in top:
                #print('{} with score {}'.format(t,document_ngrams[file][t]))
                r = r + 1
                if t in golden_standards[file]:
                    rd = r
                    break
            MRR = MRR + rd
        MRR = MRR / len(golden_standards)
        print('MRR for {}: {}'.format(k, MRR))
