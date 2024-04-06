import pickle

def save_phraser_to_txt(phraser, file_path):
    with open(file_path, 'w', encoding='utf-8') as file:
        for phrase, score in phraser.phrasegrams.items():
            # 将每个 bytes 元素解码为 str
            phrase_str = '_'.join(word.decode('utf-8') for word in phrase)
            line = f"{phrase_str}: {score}\n"
            file.write(line)

# 加载 .pkl 文件
with open('alloy2vec/training/models/model_121520_phraser.pkl', 'rb') as file:
    phraser = pickle.load(file)

# 保存 phraser 内容到 txt 文件
save_phraser_to_txt(phraser, "/home/etica/Project/alloy2vec/alloy2vec/training/models/pei_phraser_data.txt")
