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


models = [model_1989_2003, model_2004_2009, model_2010_2014, model_2015_2018, model_2019_2021, model_2022_2023]
time_periods = ['1989-2003', '2004-2009', '2010-2014', '2015-2018', '2019-2021', '2022-2023']

# 定义不同的查询词组和特定词及其简短描述
queries = [
    (['high_entropy_alloy', 'multi-principal_element_alloy', 'baseless_alloy',
      'multi-component_alloy', 'complex_concentrated_alloy', 'compositionally_complex_alloy'], 'CrFeNi', 'HEAs vs. CrFeNi'),
    (['superalloy'], 'IN718', 'Superalloy vs. IN718'),
    (['creep'], 'Sn-Ag-Cu', 'Creep vs. Sn-Ag-Cu'),
    (['neutron_irradiation', 'proton_irradiation'], 'V-TH', 'Irradiation vs. V-TH'),
    (['hydrogen_embrittlement'], 'martensitic_steels', 'HE vs. martensitic steels'),
    (['gas_pipeline'], 'WWER', 'Gas Pipeline vs. WWER')
]

# 颜色和线型设置
colors = ['red', 'blue', 'green', 'purple', 'orange', 'brown']
linestyles = ['-', '--', '-.', ':', (0, (3, 1, 1, 1)), (0, (5, 10))]

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
plt.ylim(0.2, 0.9)  # 设置y轴范围
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', title="Query vs. Target")  # 将图例放在图形外侧
plt.tight_layout()

# 图像保存路径
save_path = '/home/etica/Project/alloy2vec/visualization/relationship_between_two_words/multiple_lines.png'
plt.savefig(save_path)

plt.show()