import pandas as pd

file_path = '/home/etica/Project/alloy2vec/visualization/relationship_between_two_words/multi-component_similarity_nofilter.txt'
df = pd.read_csv(file_path)

# 计算每个材料名称出现的次数
word_counts = df['Word'].value_counts()

# 筛选至少出现4次的材料名称
words_at_least_4 = word_counts[word_counts >= 4].index.tolist()
filtered_df = df[df['Word'].isin(words_at_least_4)]

# 按“Word”分组，并聚合“Similarity”和“Time Period”的首次和最后出现
grouped = filtered_df.groupby('Word').agg({'Similarity': ['first', 'last'], 'Time Period': ['first', 'last']})

# 重新命名列，使其更易于引用
grouped.columns = ['_'.join(col).strip() for col in grouped.columns.values]

# 计算每个材料名称在“Similarity”上的变化范围（最后一次出现与第一次出现的差值）
grouped['similarity_diff'] = grouped['Similarity_last'] - grouped['Similarity_first']

# 根据'similarity_diff'降序排序结果，降序表示寻找相似度增加最多的材料
sorted_df = grouped.sort_values(by='similarity_diff', ascending=True).reset_index()

# 准备最终的DataFrame，选择需要的列
final_df = sorted_df[['Word', 'similarity_diff', 'Time Period_first', 'Time Period_last', 'Similarity_first', 'Similarity_last']]

# 如果需要，可以在这里将final_df导出到CSV或其他格式
final_df.to_csv('/home/etica/Project/alloy2vec/visualization/relationship_between_two_words/final_results_similarity_multi-component.txt', index=False)
