"""
Runs the scripts necessary to answer the question
"""
import tokenizer
import page_rank
import graph_constructor
from nltk.stem import PorterStemmer
if __name__ == '__main__':
    folderTokenizer = tokenizer.FolderTokenizer('./wwwSmall/abstracts',pathToStopWords='./stopwords.txt',wordsToKeep=['NN','NNS','NNP','NNPS','JJ'],stemmer=PorterStemmer())
    convergence = 20
    for graph in folderTokenizer.ngrams:
        page_ranking = page_rank.page_rank(folderTokenizer.ngrams[graph], 0.85, convergence)
        print('Convergence is {}'.format(convergence))
        print(page_ranking)
