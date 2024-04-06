import re

# 定义正则表达式，匹配 "high entropy alloy" 及其变体，忽略大小写
pattern = re.compile(r'high entropy alloy', re.IGNORECASE)

# 文件路径
file_path = '/home/etica/Project/alloy2vec/co_occurrence/extracted_2003_1989.txt'

try:
    # 初始化计数器
    count = 0

    # 打开并读取文件
    with open(file_path, 'r') as file:
        for line in file:
            # 在每一行中查找所有匹配项
            matches = pattern.findall(line)
            # 更新计数器
            count += len(matches)

    # 打印结果
    print(f"Total occurrences found: {count}")
except Exception as e:
    print(f"An error occurred: {e}")
