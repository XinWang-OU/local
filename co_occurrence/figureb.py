import matplotlib.pyplot as plt
import re
from collections import defaultdict

# 定义时间段和文件名
time_periods = ['1989-2003', '2004-2009', '2010-2014', '2015-2018', '2019-2021', '2022-2023']
files = {
    '1989-2003': '/home/etica/Project/alloy2vec/co_occurrence/extracted_2003_1989.txt',
    '2004-2009': '/home/etica/Project/alloy2vec/co_occurrence/extracted_2009_2004.txt',
    '2010-2014': '/home/etica/Project/alloy2vec/co_occurrence/extracted_2014_2010.txt',
    '2015-2018': '/home/etica/Project/alloy2vec/co_occurrence/extracted_2018_2015.txt',
    '2019-2021': '/home/etica/Project/alloy2vec/co_occurrence/extracted_2021_2019.txt',
    '2022-2023': '/home/etica/Project/alloy2vec/co_occurrence/extracted_2022_2023.txt'
}

# 定义关键词和相应的正则表达式
keywords = {
    'complex concentrated alloy': re.compile('complex-?\s*concentrated\s+alloy', re.IGNORECASE),
    'compositionally complex alloy': re.compile('compositionally-?\s*complex\s+alloy', re.IGNORECASE),
    'high entropy alloy': re.compile('high entropy alloy', re.IGNORECASE),
    'multi-component alloy': re.compile('multi-?\s*component\s+alloy', re.IGNORECASE),
    'multi-principal element alloy': re.compile('multi-?\s*principal\s+element\s+alloy', re.IGNORECASE),
}


# 初始化统计数据结构
counts = defaultdict(lambda: defaultdict(int))
total_lines_with_keywords = defaultdict(int)

# 处理文件和关键词
for period, file_path in files.items():
    with open(file_path, 'r') as file:
        for line in file:
            found_keyword = False
            for keyword, pattern in keywords.items():
                if pattern.search(line):
                    counts[period][keyword] += 1
                    found_keyword = True
            if found_keyword:
                total_lines_with_keywords[period] += 1

# 计算相对频率
relative_frequencies = defaultdict(lambda: defaultdict(float))
for period in time_periods:
    for keyword in keywords:
        if total_lines_with_keywords[period] > 0:
            relative_frequencies[period][keyword] = counts[period][keyword] / total_lines_with_keywords[period]

# 绘制折线图
plt.figure(figsize=(10, 10))
colors = ['blue', 'green', 'red', 'cyan', 'magenta']  # 定义颜色列表
for i, (keyword, color) in enumerate(zip(keywords, colors)):
    frequencies = [relative_frequencies[period][keyword] for period in time_periods]
    plt.plot(time_periods, frequencies, label=keyword, color=color)  # 使用color参数指定颜色

plt.xlabel('Year', fontsize=18)
plt.ylabel('Relative Frequency', fontsize=18)
# plt.title('Relative Frequency of Keywords over Time', fontsize=14)
plt.legend(fontsize=16)
plt.tick_params(axis='x', labelsize=18)  # 设置x轴刻度标签的字号为16
plt.tick_params(axis='y', labelsize=18) 
plt.tight_layout()
plt.show()