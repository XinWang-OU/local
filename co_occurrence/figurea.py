import re
import matplotlib.pyplot as plt

def count_articles_and_calculate_ratio(file_path, patterns):
    total_lines = 0
    matching_count = 0
    with open(file_path, 'r') as file:
        for line in file:
            total_lines += 1
            columns = line.split(',')  # 根据实际的列分隔符修改
            if any(re.search(pattern, column, re.IGNORECASE) for column in columns for pattern in patterns):
                matching_count += 1
    # 减1是为了排除可能的标题行
    ratio = matching_count / (total_lines - 1) if total_lines > 1 else 0
    return ratio

# 定义要查找的模式
patterns = [
    'complex-?\s*concentrated\s+alloy',
    'compositionally-?\s*complex\s+alloy',
    'high entropy alloy',
    'multi-?\s*component\s+alloy',
    'multi-?\s*principal\s+element\s+alloy'
]

# 定义时间段和对应的文件路径
periods_files = {
    '1989-2003': '/home/etica/Project/alloy2vec/co_occurrence/extracted_2003_1989.txt',
    '2004-2009': '/home/etica/Project/alloy2vec/co_occurrence/extracted_2009_2004.txt',
    '2010-2014': '/home/etica/Project/alloy2vec/co_occurrence/extracted_2014_2010.txt',
    '2015-2018': '/home/etica/Project/alloy2vec/co_occurrence/extracted_2018_2015.txt',
    '2019-2021': '/home/etica/Project/alloy2vec/co_occurrence/extracted_2021_2019.txt',
    '2022-2023': '/home/etica/Project/alloy2vec/co_occurrence/extracted_2022_2023.txt'
}

ratios = [count_articles_and_calculate_ratio(file_path, patterns) for file_path in periods_files.values()]

# 画柱状图
plt.figure(figsize=(10, 10))
plt.bar(periods_files.keys(), ratios, color='#5EC286')
plt.xlabel('Year', fontsize=18)
plt.ylabel('Ratio of Articles', fontsize=18)
plt.tick_params(axis='x', labelsize=18) 
plt.tick_params(axis='y', labelsize=18) 
# plt.title('Ratio of Articles by Year', fontsize=14)
# plt.xticks(rotation=45)
plt.show()