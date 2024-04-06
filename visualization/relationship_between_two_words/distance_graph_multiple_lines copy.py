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


# 定义不同的查询词组和特定词及其简短描述
queries = [
    (['multi-component'], 'binary', 'Binary'),
    (['multi-component'], 'ternary', 'Ternary'),
    (['multi-component'], 'quaternary', 'Quaternary'),
    (['multi-component'], 'quinary', 'Quinary'),
    (['multi-component'], 'Al-Zr', 'Al-Zr'),
    (['multi-component'], 'Fe-Al', 'Fe-Al'),
    (['multi-component'], 'Nb-Si', 'Nb-Si'),
    (['multi-component'], 'Ni-Nb', 'Ni-Nb'),
    (['multi-component'], 'high_entropY_alloys', 'high_entropy_alloy'),
]

# 颜色和线型设置
colors = ['red', 'red','blue','blue', 'green', 'green','purple','purple','black']
linestyles = ['-', '--', '-.', ':','-', '--', '-.', ':','--']

def get_average_vector(words, model):
    vectors = [model.wv[word] for word in words if word in model.wv]
    return np.mean(vectors, axis=0) if vectors else None

plt.figure(figsize=(10, 8))  # 设置图形大小

# 遍历每组查询词和特定词
for i, (query_words, target_word, label) in enumerate(queries):
    cosine_distances = []
    for model in models:
        average_vector = get_average_vector(query_words, model)
        if average_vector is not None and target_word in model.wv:
            similarity = model.wv.cosine_similarities(average_vector, [model.wv[target_word]])[0]
            distance = 1 - similarity  # 计算cosine distance
            cosine_distances.append(distance)
        else:
            cosine_distances.append(None)  # 如果查询词在模型中不存在，添加None
    # 指定颜色和线型
    plt.plot(time_periods, cosine_distances, marker='o', label=label, color=colors[i % len(colors)], linestyle=linestyles[i % len(linestyles)])

plt.xlabel('Time Period')
plt.ylabel('Average Cosine Distance')
plt.title('Change in Relationship over Time for Various Queries')
plt.xticks(rotation=45)
plt.grid(False)
plt.ylim(0.4, 0.8)  # 设置y轴范围
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', title="Query vs. Target")  # 将图例放在图形外侧
plt.tight_layout()

# 图像保存路径
save_path = '/home/etica/Project/alloy2vec/visualization/relationship_between_two_words/multiple_component.png'
plt.savefig(save_path)

plt.show()