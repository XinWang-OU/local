import os
import pickle

# 指定目录路径
directory_path = '/home/etica/Project/alloy2vec/keywords/keywords_all_year'
# 输出文件路径
output_file_path = '/home/etica/Project/alloy2vec/keywords/keywords_all_year/all_ents.p'

# 准备一个列表来收集所有单词
all_words = []

# 遍历指定目录下的所有文件
for filename in os.listdir(directory_path):
    # 检查文件扩展名是否为.txt
    if filename.endswith('.txt'):
        # 构造完整的文件路径
        file_path = os.path.join(directory_path, filename)
        # 读取并处理文件
        with open(file_path, 'r') as file:
            for line in file:
                # 分割每一行的单词
                words = line.strip().split()
                # 将单词添加到列表中
                all_words.extend(words)

# 使用pickle模块将列表序列化并写入到文件中
with open(output_file_path, 'wb') as output_file:
    pickle.dump(all_words, output_file)

print(f"所有单词已成功写入到{output_file_path}文件中。")
