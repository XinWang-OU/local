import pickle

pkl_file_path = '/home/etica/Project/alloy2vec/keywords/timeslice_keyword/all_ents_timeslice.p'

with open(pkl_file_path, 'rb') as pkl_file:
    data = pickle.load(pkl_file)

print(data)
