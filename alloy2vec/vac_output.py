from gensim.models import Word2Vec
import sys
sys.path.append('/home/etica/Project/alloy2vec')

model = Word2Vec.load("/home/etica/Project/alloy2vec/alloy2vec/training/models/model_with_wx_process_parallel")

vocab = list(model.wv.vocab)

with open("vocab.txt", "w", encoding="utf-8") as file:
    for word in vocab:
        file.write(word + "\n")
