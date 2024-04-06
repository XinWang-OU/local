import matplotlib.pyplot as plt
from gensim.models import Word2Vec
import numpy as np
import os
import sys
sys.path.append('/home/etica/Project/alloy2vec')

model_1989_2003 = Word2Vec.load("/home/etica/Project/alloy2vec/alloy2vec/training/models/aligned_timeslice_models/aligned_1model_2003_1989")
model_2004_2009 = Word2Vec.load("/home/etica/Project/alloy2vec/alloy2vec/training/models/aligned_timeslice_models/aligned_1model_2009_2004")
model_2010_2014 = Word2Vec.load("/home/etica/Project/alloy2vec/alloy2vec/training/models/aligned_timeslice_models/aligned_1model_2014_2010")
model_2015_2018 = Word2Vec.load("/home/etica/Project/alloy2vec/alloy2vec/training/models/aligned_timeslice_models/aligned_1model_2018_2015")
model_2019_2021 = Word2Vec.load("/home/etica/Project/alloy2vec/alloy2vec/training/models/aligned_timeslice_models/aligned_1model_2021_2019")
model_2022_2023 = Word2Vec.load("/home/etica/Project/alloy2vec/alloy2vec/training/models/aligned_timeslice_models/aligned_1model_2022_2023")


models = [model_1989_2003, model_2004_2009, model_2010_2014, model_2015_2018, model_2019_2021, model_2022_2023]
time_periods = ['1989-2003', '2004-2009', '2010-2014', '2015-2018', '2019-2021', '2022-2023']

query_words = ['superalloy']

def get_average_vector(words, model):
    vectors = [model.wv[word] for word in words if word in model.wv]
    return np.mean(vectors, axis=0) if vectors else None

cosine_distances = []
for model in models:
    average_vector = get_average_vector(query_words, model)
    if average_vector is not None:
        similarity = model.wv.cosine_similarities(average_vector, [model.wv['IN718']])[0]
        distance = 1 - similarity  # 计算cosine distance
        cosine_distances.append(distance)
    else:
        cosine_distances.append(None)  # 如果查询词在模型中不存在，添加None

plt.plot(time_periods, cosine_distances, marker='o')
plt.xlabel('Time Period')
plt.ylabel('Average Cosine Distance with IN718')
plt.title('Change in Relationship over Time')
plt.xticks(rotation=45)
plt.grid(True)

output_dir = '/home/etica/Project/alloy2vec/visualization/relationship_between_two_words'
file_name = '_'.join(query_words) + 'VS_UNS_S30403_similarity.png'
file_path = os.path.join(output_dir, file_name)

# 保存图片
plt.savefig(file_path)
plt.show()