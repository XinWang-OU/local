import pickle

pkl_file_path = '/home/etica/Project/alloy2vec/keywords/keywords_all_year/all_ents.p'

with open(pkl_file_path, 'rb') as pkl_file:
    data = pickle.load(pkl_file)

print(data)
