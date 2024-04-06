import matplotlib.pyplot as plt
from gensim.models import Word2Vec
import sys

sys.path.append('/home/etica/Project/alloy2vec')

model_1989_2003 = Word2Vec.load("/home/etica/Project/alloy2vec/alloy2vec/training/models/aligned_timeslice_models/aligned_1model_2003_1989")
model_2004_2009 = Word2Vec.load("/home/etica/Project/alloy2vec/alloy2vec/training/models/aligned_timeslice_models/aligned_1model_2009_2004")
model_2010_2014 = Word2Vec.load("/home/etica/Project/alloy2vec/alloy2vec/training/models/aligned_timeslice_models/aligned_1model_2014_2010")
model_2015_2018 = Word2Vec.load("/home/etica/Project/alloy2vec/alloy2vec/training/models/aligned_timeslice_models/aligned_1model_2018_2015")
model_2019_2021 = Word2Vec.load("/home/etica/Project/alloy2vec/alloy2vec/training/models/aligned_timeslice_models/aligned_1model_2021_2019")
model_2022_2023 = Word2Vec.load("/home/etica/Project/alloy2vec/alloy2vec/training/models/aligned_timeslice_models/aligned_1model_2022_2023")

# 将模型放入一个列表
models = [model_1989_2003, model_2004_2009, model_2010_2014, model_2015_2018, model_2019_2021, model_2022_2023]
time_periods = ['1989-2003','2004-2009','2010-2014','2015-2018','2019-2021','2022-2023']
key_word = 'multi-component'  

# 用于存储每个时间段最相近的词
closest_words_per_period = {}

# 初始化一个变量来存储最大距离
max_distance = 0

# Iterate over each model and time period
for model, time in zip(models, time_periods):
    try:
        # Attempt to find the most similar words to the key_word
        closest_words = model.wv.most_similar(positive=[key_word], topn=10)
        closest_words_per_period[time] = closest_words
        # 更新最大距离
        max_distance = max(max_distance, max([1 - similarity for _, similarity in closest_words]))
    except KeyError:
        # If the key_word is not found in the model's vocabulary, skip or mark as N/A
        closest_words_per_period[time] = [('N/A', 0)]
print(closest_words_per_period)

# 绘制图形
fig, ax = plt.subplots()

# 添加每个词到图形
for time, closest_words in closest_words_per_period.items():
    for word, similarity in closest_words:
        x = time_periods.index(time)
        y = 1 - similarity  # 使用1减去相似度以得到距离
        ax.text(x, y, word, ha='center', va='bottom')

# 设置图形属性
ax.set_xticks(range(len(time_periods)))
ax.set_xticklabels(time_periods)
ax.set_title(f'Word Similarity Evolution for "{key_word}"')
ax.set_ylabel('Distance from Key Word')
ax.set_xlabel('Time Period')

# 根据最大距离设置纵轴的范围
ax.set_ylim([0, max_distance + 0.05])  # 添加一点额外空间以确保文本可见

# 保存图形到指定的文件夹
output_path = '/home/etica/Project/alloy2vec/visualization/similarity_evolution.png'
plt.savefig(output_path)
print(f"Plot saved to {output_path}")

# 显示图形（如果你在Jupyter Notebook等环境中运行，你也可以选择显示出来）
plt.show()
