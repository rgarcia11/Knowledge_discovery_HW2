# Knowledge_discovery_HW2

Second homework of the Knowledge Discovery from Social and Information Networks summer class in Universidad de los Andes.
This work is presented by Rogelio Garcia. It is available at the course's virtual machine and at Sicuaplus.
It was solved in Python3.3.

## Questions
### 1.
**Write a program that loads each document into a word graph (either directed or undirected). In doing so, tokenize on whitespace, remove stopwords, and keep only the nouns and adjectives corresponding to {NN, NNS, NNP, NNPS, JJ}. Apply a stemmer on every word. For each candidate words, create a node in the graph. Add an edge in the graph between two words if they are adjacent in the original text. The weight wij of an edge (vi, vj ) is calculated as the number of times the corresponding words wi and wj are adjacent in text.**

### 2.
**Run PageRank on each word graph corresponding to each document in the collection as follows:**

**• Initialization: s = [s(v1), · · · , s(vn)] = [ 1 n , · · · , 1 n ], where n = |V |.**

**• Score nodes in a graph using their PageRank obtained by recursively computing the equation:**

> s(vi) = α * sum(vj∈Adj(vi), wji * s(vj) / sum(vk∈Adj(vj), wjk)) + (1 − α) * pi,

  **where α is a damping factor (α = 0.85) and p = [ 1n, · · · ,1n].**
  
*Bonus credit if the above formula is used as is, instead of simplifying wjk and wjk as 1 and using the below formula instead:*

> s(vi) = α * sum(vj∈Adj(vi), s(vj) / out(vj)) + (1 − α) * pi,

*where out(vj) is the outdegree of the node vj.*

### 3.
**After the PageRank convergence or a fixed number of iterations is reached, form n-grams of length up to 3 (unigrams, bigrams and trigrams) from words adjacent in text and score n-grams or phrases using the sum of scores of individual words that comprise the phrase.**
### 4.
**Calculate the MRR for the entire collection for top-k ranked n-grams or phrases, where k ranges from 1 to 10, as follows, using the gold-standard (author annotated data) provided in sicua:**
**• Mean reciprocal rank, MRR**

> MRR = 1/|D| * sum(d=1,|D|,1/rd)

*where rd is the rank at which the first correct prediction was found for d ∈ D.*

### 5. (Bonus credit)
***[Extra-credit - 50 points]* Compare the MRR of the above PageRank algorithm with the MRR of a ranking of words based on their TF-IDF ranking scheme. Calculate the TF component from each document and the IDF component from the entire collection.**

## Solution

## How to run

## Limitations
Some phrases in the golden standard are beyond unigram, bigram and trigrams, as they could be comprised of longer sentences. For example, file 2466 has only one sentence, "content analysis and indexing", that doesn't match any possible output of the program, as it could only output trigrams, at most. Even if "analysis and indexing" or "content analysis" is an output, it would not be registered as a correct match.
