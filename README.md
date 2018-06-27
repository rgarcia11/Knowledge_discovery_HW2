# Knowledge_discovery_HW2

Second homework of the Knowledge Discovery from Social and Information Networks summer class in Universidad de los Andes.
This work is presented by Rogelio Garcia. It is available at the course's virtual machine and at Sicuaplus.
It was solved in Python3.3.

## Solution to each question
For these questions, examples of each step are provided. For this, a reduced version of file '183' is used, located in the folder wwwSmall/abstracts:
The_DT Eigentrust_NNP algorithm_NN for_IN reputation_NN management_NN in_IN P2P_NN networks_NNS

With its golden standard in wwwSmall/gold:
distributed eigenvector computation
peer-to-peer
reputation

### 1.
**Write a program that loads each document into a word graph (either directed or undirected). In doing so, tokenize on whitespace, remove stopwords, and keep only the nouns and adjectives corresponding to {NN, NNS, NNP, NNPS, JJ}. Apply a stemmer on every word. For each candidate words, create a node in the graph. Add an edge in the graph between two words if they are adjacent in the original text. The weight wij of an edge (vi, vj ) is calculated as the number of times the corresponding words wi and wj are adjacent in text.**

For this point, the script *tokenizer.py* is used, particularly the class **FolderTokenizer**. That class is initialized with the following parameters: Path to the files to tokenize, path to stop words list, words to keep according to the notation in the exercise (in this case nouns and adjectives) and a stemmer. The following code preview shows the necessary imports and the calls to obtain the tokenized folder.
```python
import tokenizer
from nltk.stem import PorterStemmer

tokenizedFolder = tokenizer.FolderTokenizer('./www/abstracts',pathToStopWords='./stopwords.txt',wordsToKeep=['NN','NNS','NNP','NNPS','JJ'],stemmer=PorterStemmer())

folderTokenizer = tokenizedFolder.ngrams
```
The *ngrams* attribute of the FolderTokenizer class contains the graph. This class assumes n=1 for the ngrams relation, meaning it is taking adjacent words. The format is a dictionary of graphs. The dictionary's key is the name of the file and its value is a graph object (an object of the Undirected_graph class in the graph_constructor file). This graph object is a dictionary with each token being a key, and the value is another dictionary where each key is an adjacent token, and the value is the number of times both words are adjacent.

```javascript
'183':<graph_constructor.Undirected_graph object at 0x0A167B50>
```
```javascript
{'183': {'eigentrust': 0.408248290463863, 'algorithm': 0.408248290463863, 'reput': 0.408248290463863, 'manag': 0.408248290463863, 'p2p': 0.408248290463863, 'network': 0.408248290463863}}
```

### 2.
**Run PageRank on each word graph corresponding to each document in the collection as follows:**

**• Initialization: s = [s(v1), · · · , s(vn)] = [ 1 n , · · · , 1 n ], where n = |V |.**

**• Score nodes in a graph using their PageRank obtained by recursively computing the equation:**

> s(vi) = α * sum(vj∈Adj(vi), wji * s(vj) / sum(vk∈Adj(vj), wjk)) + (1 − α) * pi,

  **where α is a damping factor (α = 0.85) and p = [ 1n, · · · ,1n].**
  
*Bonus credit if the above formula is used as is, instead of simplifying wjk and wjk as 1 and using the below formula instead:*

> s(vi) = α * sum(vj∈Adj(vi), s(vj) / out(vj)) + (1 − α) * pi,

*where out(vj) is the outdegree of the node vj.*

This part uses the *page_rank.py* script. This script only contains two functions, one to calculate the above formula for s, and another one to compute this formula for each element of the graph, several times as specified by parameter. It's used like this in the script:

```python
import page_rank
document_scores = {}
convergence = 20
for graph in folderTokenizer:
    document_scores[graph] = page_rank.page_rank(folderTokenizer[graph], 0.85, convergence)
```
Convergence is the numbero of times the algorithm will be run for all nodes in the graph.

The dictionary **document_scores** will be in the format *filename:{word:score}*. The following is a snippet showing all word scores for the file '183'.
```javascript
{'183': {'eigentrust': 0.408248290463863, 'algorithm': 0.408248290463863, 'reput': 0.408248290463863, 'manag': 0.408248290463863, 'p2p': 0.408248290463863, 'network': 0.408248290463863}}
```
Scores are normalized and words are stemmed.
### 3.
**After the PageRank convergence or a fixed number of iterations is reached, form n-grams of length up to 3 (unigrams, bigrams and trigrams) from words adjacent in text and score n-grams or phrases using the sum of scores of individual words that comprise the phrase.**
This part uses the script *tokenizer.py* again, but the class **nGramTokenizer**. This class works the same way as the class used in point 1, but its implementation is a lot simpler since it doesn't build a graph of adjacency nor does it remove stopwords and has fewer considerations. The ngrams for n=1 (unigrams), 2 (bigrams) and 3 (trigrams) are built for this exercise. The class receives the path to the files, the stemmer to be used, and this time we use the n parameter, which is the length of the ngram.
```python
import tokenizer
document_multingrams = {}
for i in range(1,4):
    document_multingrams[i] =  tokenizer.nGramTokenizer('./www/abstracts',stemmer=PorterStemmer(),n=i).ngrams
```
The object **document_multigrams** is a dictionary with key being the length of the ngram and value being a second dictionary. The second dictionary has key: filename and value: ngram, where ngram is a third dictionary of each ngram and the value 0. This value is where the scores for the ngram will be calculated and saved later. The following shows an example of the dictionaries for file 183 (example was cut this time).
```javascript
{1: {'183': {'the': 0, 'eigentrust': 0, 'algorithm': 0, 'for': 0, 'reput': 0, 'manag': 0, 'in': 0, 'p2p': 0, 'network': 0}}, 
2: {'183': {'the eigentrust': 0, 'eigentrust algorithm': 0, 'algorithm for': 0, 'for reput': 0, 'reput manag': 0, 'manag in': 0, 'in p2p': 0, 'p2p network': 0}}, 
3: {'183': {'the eigentrust algorithm': 0, 'eigentrust algorithm for': 0, 'algorithm for reput': 0, 'for reput manag': 0, 'reput manag in': 0, 'manag in p2p': 0, 'in p2p network': 0}}}
```

The next part joins all the ngrams in one single dictionary **document_ngrams** and sums up the value of each word in the ngram.
```python
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
```
The result of the earlier example is this:
```javascript
{'183': {'the': 0, 'eigentrust': 0.408248290463863, 'algorithm': 0.408248290463863, 'for': 0, 'reput': 0.408248290463863, 'manag': 0.408248290463863, 'in': 0, 'p2p': 0.408248290463863, 'network': 0.408248290463863, 'the eigentrust': 0.408248290463863, 'eigentrust algorithm': 0.816496580927726, 'algorithm for': 0.408248290463863, 'for reput': 0.408248290463863, 'reput manag': 0.816496580927726, 'manag in': 0.408248290463863, 'in p2p': 0.408248290463863, 'p2p network': 0.816496580927726, 'the eigentrust algorithm': 0.816496580927726, 'eigentrust algorithm for': 0.816496580927726, 'algorithm for reput': 0.816496580927726, 'for reput manag': 0.816496580927726, 'reput manag in': 0.816496580927726, 'manag in p2p': 0.816496580927726, 'in p2p network': 0.816496580927726}}
```

### 4.
**Calculate the MRR for the entire collection for top-k ranked n-grams or phrases, where k ranges from 1 to 10, as follows, using the gold-standard (author annotated data) provided in sicua:**
**• Mean reciprocal rank, MRR**

> MRR = 1/|D| * sum(d=1,|D|,1/rd)

*where rd is the rank at which the first correct prediction was found for d ∈ D.*

This part uses a different kind of tokenizer, also from the *tokenizer.py* script. This tokenizer takes each line as a token and stems it with the given stemmer.

```python
import tokenizer
golden_standards = tokenizer.lineTokenizer('./wwwSmall/gold',stemmer=PorterStemmer()).ngrams
```

The output being a dictionaroy with key: filename, value: a dictionary with key: line and value: 0. The value is 0 because it won't be used, and a dictionary is used to use simpler Python expressions later on.
```javascript
{'183': {'distribut eigenvector comput': 0, 'peer-to-p': 0, 'reput': 0}}
```
The next part iterates through a set of *k* values as per the instructions. Within each *k-iteration*, the MRR for each file is calculated using the top k-ranked words for that particular file, and figuring out the position in the rank of the first match with the golden standard. It then prints out the result for each *k* and writes it to a file that is used to generate a diagram to compare TF_IDF with page rank.
```python
import tokenizer
from heapq import nlargest

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
```
MRR is built with the following format:
```javascript
{1: 0.0, 2: 0.0, 3: 0.0, 4: 0.0, 5: 0.0, 6: 0.0, 7: 0.0, 8: 0.0, 9: 0.0, 10: 0.0}
```
In this case everything is 0 because of the top 10 words non is in the golden standard. Top 10:
```javascript
['eigentrust algorithm', 'reput manag', 'p2p network', 'the eigentrust algorithm', 'eigentrust algorithm for', 'algorithm for reput', 'for reput manag', 'reput manag in', 'manag in p2p', 'in p2p network']
```

### 5. (Bonus credit)
***[Extra-credit - 50 points]* Compare the MRR of the above PageRank algorithm with the MRR of a ranking of words based on their TF-IDF ranking scheme. Calculate the TF component from each document and the IDF component from the entire collection.**

This part does the same as the last part, but the input is built anew. Before doing the same process to compute the MRR as before, the TF-IDF representation must be built. First, we use again the first tokenizer we built.
```python
import tokenizer
from nltk.stem import PorterStemmer

tokenizedFolder = tokenizer.FolderTokenizer('./www/abstracts',pathToStopWords='./stopwords.txt',wordsToKeep=['NN','NNS','NNP','NNPS','JJ'],stemmer=PorterStemmer())
```
The **vocabulary** attribute contains the count of each word over each file, so that is the term frequency *tf*. The **wordCount** attribute contains the total count over all files, so that is the document frequency *df*. The inverted document frequency is then calculated as the *log(#OfWords/df)*, where log is the base-2-logarithm and #OfWords is the size of the vocabulary. After that, the multiplication of *tf* and *idf* is taken. The term frequency *tf* is normalized before multiplicating: each *tf* is divided by the maximum frequency *maxf*.
```python
import math
from heapq import nlargest

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
```
Dictionary *idf* before before calculating the logarithm (so, only the *df* part):
```javascript
{'eigentrust': 1, 'algorithm': 1, 'reput': 1, 'manag': 1, 'p2p': 1, 'network': 1}
```
Dictionary *tf* before normalizing:
```javascript
{'183': {'eigentrust': 1, 'algorithm': 1, 'reput': 1, 'manag': 1, 'p2p': 1, 'network': 1}}
```
Dictionary *tf* after normalization (the only difference is that, as it was divided by 1, the value doesn't change but the type changes from int to double):
```javascript
{'183': {'eigentrust': 1.0, 'algorithm': 1.0, 'reput': 1.0, 'manag': 1.0, 'p2p': 1.0, 'network': 1.0}}
```

The result is a dictionary with key: filename and value: another dictionary with key: word and value: tf-idf of that word in that document. Example below:
```javascript
{'183': {'eigentrust': 0.0, 'algorithm': 0.0, 'reput': 0.0, 'manag': 0.0, 'p2p': 0.0, 'network': 0.0}}
```
Keep in mind that this example, as it only uses 1 document, will have all zero values as each words appears in all documents, therefore *df=1* and *len(tf)* is 1 also. Log of 1 is 0.

The next part is essentially the same as in point 4. The MRR is calculated.
```python
from heapq import nlargest

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
```
The format is the same as in point 4. The 10 largest words retrieved by the function nlargest are:
```javascript
['eigentrust', 'algorithm', 'reput', 'manag', 'p2p', 'network']
```
As reputation is present in the golden standard, and reput is the third word in the ranking, MRR is:
```javascript
{1: 0.0, 2: 0.0, 3: 0.3333333333333333, 4: 0.3333333333333333, 5: 0.3333333333333333, 6: 0.3333333333333333, 7: 0.3333333333333333, 8: 0.3333333333333333, 9: 0.3333333333333333, 10: 0.3333333333333333}
```
When the code is run with the correct folder /www/abstracts and /www/gold, the output is as follows:
```
1. Creating the graph.
2. Calculating Page Rank.
3. Building ngrams and calculating score.
4. Calculating the MRR for the Page Rank method.
MRR when looking at the top 1 words: 0.04887218045112782
MRR when looking at the top 2 words: 0.07894736842105263
MRR when looking at the top 3 words: 0.0987468671679197
MRR when looking at the top 4 words: 0.1156641604010025
MRR when looking at the top 5 words: 0.12679197994987457
MRR when looking at the top 6 words: 0.13406015037593969
MRR when looking at the top 7 words: 0.1383566058002146
MRR when looking at the top 8 words: 0.1414581095596131
MRR when looking at the top 9 words: 0.14363020646855215
MRR when looking at the top 10 words: 0.1457354696264468
5. Comparing with TF-IDF.
MRR for TF-IDF when looking at the top 1 words: 0.10225563909774436
MRR for TF-IDF when looking at the top 2 words: 0.13383458646616542
MRR for TF-IDF when looking at the top 3 words: 0.14962406015037605
MRR for TF-IDF when looking at the top 4 words: 0.15620300751879712
MRR for TF-IDF when looking at the top 5 words: 0.16131578947368422
MRR for TF-IDF when looking at the top 6 words: 0.16494987468671674
MRR for TF-IDF when looking at the top 7 words: 0.16763515932688855
MRR for TF-IDF when looking at the top 8 words: 0.16942087361260286
MRR for TF-IDF when looking at the top 9 words: 0.17075754863348844
MRR for TF-IDF when looking at the top 10 words: 0.17165980427258612
```
The comparison between the two was made using a Jupyter Notebook that only reads the two files that were written and creates a diagram. It uses the Jupyter built in *%pylab inline* dependency.

*In [1]:*
<pre>
%pylab inline
</pre>
*Out [1]:*
<pre>
Populating the interactive namespace from numpy and matplotlib
</pre>
*In [2]:*
<pre>
MRR_Scores = loadtxt('MRR.txt')
MRR_TF_IDF_Scores = loadtxt('MRR_tf_idf.txt')
k = linspace(1,10,10)
plot(k,MRR_Scores,'-o',label='Page Rank')
plot(k,MRR_TF_IDF_Scores,'-o',label='TF-IDF')
xlabel('k')
ylabel('MRR')
legend(loc='best')
savefig('Comparison.jpg')
</pre>
*Out [2]*

![Comparison](https://i.imgur.com/dTAYccA.jpg)

## Code
The code is fully available in Sicua and the Virtual Machine.
There are four scripts: *graph_constructor.py*, *page_rank.py*, *hw2.py*, *tokenizer.py*. The script *hw2.py* runs everything, *graph_constructor.py* creates a graph and contains utilities to add a node, an edge, retrieve the weight of an edge or retrieve the whole graph. *page_rank.py* runs the page rank algorithm, and *tokenizer.py* contains 3 types of tokenizers for each step of the homework, one to tokenize everything and built an adjacency graph, another one to tokenize every line as a token on its own and another one to only build ngrams.

## How to run
It's only necessary to run *hw.py* with the following command:

> python3 hw2.py

or

> python hw2.py

And in the Virtual Machine, the script will be found in */home/estudiante/Documents/Knowledge_discovery_HW1*, And it can be reached with the following command: 

> cd /home/estudiante/Documents/Knowledge_discovery_HW1

For the Jupyter Notebook, it can be run with the following command:

> jupyter notebook

And in the web interface, look for Comparison.ipynb and run the two cells with shift-enter. This cannot be done in the Virtual Machine. However, the purpose of the notebook is only to visualize the diagram and compare the two methods, and the diagram can be found in Comparison.jpg anyways.

## Limitations
Some phrases in the golden standard are beyond unigram, bigram and trigrams, as they could be comprised of longer sentences. For example, file 2466 has only one sentence, "content analysis and indexing", that doesn't match any possible output of the program, as it could only output trigrams, at most. Even if "analysis and indexing" or "content analysis" is an output, it would not be registered as a correct match.
