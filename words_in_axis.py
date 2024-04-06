from gensim.models import Word2Vec
import numpy as np
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

# 假设model是您的Word2Vec模型
model = Word2Vec.load('/home/etica/Project/alloy2vec/alloy2vec/training/models/aligned_timeslice_models/aligned_1model_2018_2015')

similar_words = model.wv.most_similar('superalloy', topn=50)
words = ['superalloy'] + [word for word, _ in similar_words]
vectors = np.array([model.wv[word] for word in words])

# 使用t-SNE进行降维
tsne = TSNE(n_components=2, random_state=0, perplexity=30)  # 根据样本量调整perplexity
vectors_2d = tsne.fit_transform(vectors)

# 可视化
plt.figure(figsize=(20, 12))
for i, word in enumerate(words):
    plt.scatter(vectors_2d[i, 0], vectors_2d[i, 1])
    plt.annotate(word, (vectors_2d[i, 0], vectors_2d[i, 1]), textcoords="offset points", xytext=(0,10), ha='center')
plt.xlabel('Dimension 1')
plt.ylabel('Dimension 2')
plt.title('2D visualization of "superalloy" and its 100 most similar words')
plt.show()