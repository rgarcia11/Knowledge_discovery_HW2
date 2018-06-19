"""
Runs the scripts necessary to answer the question
"""
import tokenizer
import page_rank
import graph_constructor
from nltk.stem import PorterStemmer
if __name__ == '__main__':
    folderTokenizer = tokenizer.FolderTokenizer('./wwwSmall/abstracts',pathToStopWords='./stopwords.txt',wordsToKeep=['NN','NNS','NNP','NNPS','JJ'],stemmer=PorterStemmer())
    print(folderTokenizer.ngrams)
    print(folderTokenizer.ngrams['183'].graph)
    convergence = 0
    for graph in folderTokenizer.ngrams:
        page_ranking = page_rank.page_rank(folderTokenizer.ngrams[graph], 0.85, convergence=2)
        print(page_ranking)
    #page_ranking0 = page_rank.page_rank(folderTokenizer.ngram, 0.85, convergence=0)
    #print(page_ranking0)
    #page_ranking1 = page_rank.page_rank(folderTokenizer.ngram, 0.85, convergence=1)
    #print(page_ranking1)
    #page_ranking2 = page_rank.page_rank(folderTokenizer.ngram, 0.85, convergence=2)
    #print(page_ranking2)
    #page_ranking3 = page_rank.page_rank(folderTokenizer.ngram, 0.85, convergence=3)
    #print(page_ranking3)
    #page_ranking4 = page_rank.page_rank(folderTokenizer.ngram, 0.85, convergence=4)
    #print(page_ranking4)
    #page_ranking5 = page_rank.page_rank(folderTokenizer.ngram, 0.85, convergence=5)
    #print(page_ranking5)
