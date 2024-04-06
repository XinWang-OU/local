use_full_file_alloy = True
use_full_file_thermal_properties = True
use_full_file_mechanical_properties = True
use_full_file_process_general = True
use_full_file_process_melt_ded = True
use_full_process_melt_general= True
use_full_process_melt_pbf = True
use_full_process_solid_binder = True
use_full_process_solid_cold_spray = True
use_full_process_solid_extrusion = True
use_full_process_solid_field_assisted = True
use_full_process_solid_friction_based = True
use_full_process_solid_general = True

import numpy as np
from gensim import matutils

from gensim.models import Word2Vec 
w2v_model =Word2Vec.load("alloy2vec/training/models/model_with_wx_process_parallel_all") 

def predict_combined_context(w2v_model, words, topn=5000):
    vecs = [w2v_model.wv[word] for word in words if word in w2v_model.wv]
    if len(vecs) == 0:
        return []

    combined_vec = np.mean(vecs, axis=0)
    prob_values = np.exp(np.dot(combined_vec, w2v_model.trainables.syn1neg.T))
    prob_values /= np.sum(prob_values)
    top_indices = matutils.argsort(prob_values, topn=topn, reverse=True)
    return [(w2v_model.wv.index2word[index], prob_values[index]) for index in top_indices]

words_to_predict = ["steel", "annealing"]
top_context_words = predict_combined_context(w2v_model, words_to_predict, topn=500)
#top_context_words_string = '\n'.join(f"{word}: {similarity}" for word, similarity in top_context_words)

def read_keywords(file_paths, use_full):
    words_set = set()
    for file_path in file_paths: 
        with open(file_path, 'r') as file:
            for line in file:
                columns = line.strip().split('\t')
                # 总是将第一列加入集合
                words_set.add(columns[0])
                # 如果 use_full 为 True 并且存在第二列
                if use_full and len(columns) > 1:
                    # 将第二列中的单词（以空格分隔）加入集合
                    words_set.update(columns[1].split())
    return list(words_set)


# 使用新函数和独立的开关读取关键词列表
#processed_alloy_words = read_keywords('/home/etica/Project/alloy2vec/keywords/similar_alloy.txt', use_full_file_alloy)
#thermal_properties_words = read_keywords('/home/etica/Project/alloy2vec/keywords/similar_thermal_properties.txt', use_full_file_thermal_properties)
#mechanical_properties_words = read_keywords('/home/etica/Project/alloy2vec/keywords/similar_mechanical_properties.txt', use_full_file_mechanical_properties)

alloy_files = [
    '/home/etica/Project/alloy2vec/new_keyword/similar_AA_family_all.txt',
    '/home/etica/Project/alloy2vec/new_keyword/similar_alloy.txt'
]
alloy_words = read_keywords(alloy_files, use_full_file_alloy)

thermal_properties_file = ['/home/etica/Project/alloy2vec/new_keyword/similar_thermal_properties.txt']
thermal_properties_words = read_keywords(thermal_properties_file, use_full_file_thermal_properties)

mechanical_properties_file = ['/home/etica/Project/alloy2vec/new_keyword/mechanical_properties.txt']
mechanical_properties_words = read_keywords(mechanical_properties_file, use_full_file_mechanical_properties)

#process words
process_general_file = ['/home/etica/Project/alloy2vec/new_keyword/similar_process_general.txt']
process_general_words = read_keywords(process_general_file, use_full_file_process_general)

process_melt_ded_file = ['/home/etica/Project/alloy2vec/new_keyword/similar_process_melt_ded.txt']
process_melt_ded_words = read_keywords(process_melt_ded_file, use_full_file_process_melt_ded)

process_melt_general_file = ['/home/etica/Project/alloy2vec/new_keyword/similar_process_melt_general.txt']
process_melt_pbf_file = ['/home/etica/Project/alloy2vec/new_keyword/similar_process_melt_pbf.txt']
process_solid_binder_file = ['/home/etica/Project/alloy2vec/new_keyword/similar_process_solid_binder.txt']
process_solid_cold_spray_file = ['/home/etica/Project/alloy2vec/new_keyword/similar_process_solid_cold_spray.txt']
process_solid_extrusion_file = ['/home/etica/Project/alloy2vec/new_keyword/similar_process_solid_extrusion.txt']
process_solid_field_assisted_file = ['/home/etica/Project/alloy2vec/new_keyword/similar_process_solid_field_assisted.txt']
process_solid_friction_based_file = ['/home/etica/Project/alloy2vec/new_keyword/similar_process_solid_friction_based.txt']
process_solid_general_file = ['/home/etica/Project/alloy2vec/new_keyword/similar_process_solid_general.txt']

process_melt_general_words = read_keywords(process_melt_general_file, use_full_process_melt_general)
process_melt_pbf_words = read_keywords(process_melt_pbf_file, use_full_process_melt_pbf)
process_solid_binder_words = read_keywords(process_solid_binder_file, use_full_process_solid_binder)
process_solid_cold_spray_words = read_keywords(process_solid_cold_spray_file, use_full_process_solid_cold_spray)
process_solid_extrusion_words = read_keywords(process_solid_extrusion_file, use_full_process_solid_extrusion)
process_solid_field_assisted_words = read_keywords(process_solid_field_assisted_file, use_full_process_solid_field_assisted)
process_solid_friction_based_words = read_keywords(process_solid_friction_based_file, use_full_process_solid_friction_based)
process_solid_general_words = read_keywords(process_solid_general_file, use_full_process_solid_general)

ELEMENTS = ["H", "He", "Li", "Be", "B", "C", "N", "O", "F", "Ne", "Na", "Mg", "Al", "Si", "P", "S", "Cl", "Ar", "K",
            "Ca", "Sc", "Ti", "V", "Cr", "Mn", "Fe", "Co", "Ni", "Cu", "Zn", "Ga", "Ge", "As", "Se", "Br", "Kr",
            "Rb", "Sr", "Y", "Zr", "Nb", "Mo", "Tc", "Ru", "Rh", "Pd", "Ag", "Cd", "In", "Sn", "Sb", "Te", "I",
            "Xe", "Cs", "Ba", "La", "Ce", "Pr", "Nd", "Pm", "Sm", "Eu", "Gd", "Tb", "Dy", "Ho", "Er", "Tm", "Yb",
            "Lu", "Hf", "Ta", "W", "Re", "Os", "Ir", "Pt", "Au", "Hg", "Tl", "Pb", "Bi", "Po", "At", "Rn", "Fr",
            "Ra", "Ac", "Th", "Pa", "U", "Np", "Pu", "Am", "Cm", "Bk", "Cf", "Es", "Fm", "Md", "No", "Lr", "Rf",
            "Db", "Sg", "Bh", "Hs", "Mt", "Ds", "Rg", "Cn", "Nh", "Fl", "Mc", "Lv", "Ts", "Og", "Uue"]

ELEMENT_NAMES = ["hydrogen", "helium", "lithium", "beryllium", "boron", "carbon", "nitrogen", "oxygen", "fluorine",
                 "neon", "sodium", "magnesium", "aluminum", "silicon", "phosphorus", "sulfur", "chlorine", "argon",
                 "potassium", "calcium", "scandium", "titanium", "vanadium", "chromium", "manganese", "iron",
                 "cobalt", "nickel", "copper", "zinc", "gallium", "germanium", "arsenic", "selenium", "bromine",
                 "krypton", "rubidium", "strontium", "yttrium", "zirconium", "niobium", "molybdenum", "technetium",
                 "ruthenium", "rhodium", "palladium", "silver", "cadmium", "indium", "tin", "antimony", "tellurium",
                 "iodine", "xenon", "cesium", "barium", "lanthanum", "cerium", "praseodymium", "neodymium",
                 "promethium", "samarium", "europium", "gadolinium", "terbium", "dysprosium", "holmium", "erbium",
                 "thulium", "ytterbium", "lutetium", "hafnium", "tantalum", "tungsten", "rhenium", "osmium",
                 "iridium", "platinum", "gold", "mercury", "thallium", "lead", "bismuth", "polonium", "astatine",
                 "radon", "francium", "radium", "actinium", "thorium", "protactinium", "uranium", "neptunium",
                 "plutonium", "americium", "curium", "berkelium", "californium", "einsteinium", "fermium",
                 "mendelevium", "nobelium", "lawrencium", "rutherfordium", "dubnium", "seaborgium", "bohrium",
                 "hassium", "meitnerium", "darmstadtium", "roentgenium", "copernicium", "nihonium", "flerovium",
                 "moscovium", "livermorium", "tennessine", "oganesson", "ununennium"]


metal = []
thermal_property = []
mechanical_property = []
process_general = []
process_melt_ded = []
process_melt_general= []
process_melt_pbf = []
process_solid_binder = []
process_solid_cold_spray = []
process_solid_extrusion = []
process_solid_field_assisted = []
process_solid_friction_based = []
process_solid_general = []

unfiltered_item = []


for item, similarity in top_context_words:
    # Find the item and its similarity in model_results
    similarity = next((sim for word, sim in top_context_words if word == item), None)
    # Step 1: Keep items that contain process keywords
    if any(word in item for word in process_general_words):
        process_general.append((item, similarity))
    elif any(word in item for word in process_melt_ded_words):
        process_melt_ded.append((item, similarity))
    elif any(word in item for word in process_melt_general_words):
        process_melt_general.append((item, similarity))
    elif any(word in item for word in process_melt_pbf_words):
        process_melt_pbf.append((item, similarity))
    elif any(word in item for word in process_solid_binder_words):
        process_solid_binder.append((item, similarity))
    elif any(word in item for word in process_solid_cold_spray_words):
        process_solid_cold_spray.append((item, similarity))
    elif any(word in item for word in process_solid_extrusion_words):
        process_solid_extrusion.append((item, similarity))
    elif any(word in item for word in process_solid_field_assisted_words):
        process_solid_field_assisted.append((item, similarity))
    elif any(word in item for word in process_solid_friction_based_words):
        process_solid_friction_based.append((item, similarity))
    elif any(word in item for word in process_solid_general_words):
        process_solid_general.append((item, similarity))
    # Step 2: Keep items which contain elements (partial match)
    elif any(element in item for element in ELEMENTS) or item in ELEMENT_NAMES:
        metal.append((item, similarity))
    # Step 3: Keep metal items that contain any word from alloy words (full match)
    elif item in alloy_words:
        metal.append((item, similarity))
    # Step 3: Keep metal items that contain both letters and digits （是否需要？）
    elif any(char.isalpha() for char in item) and any(char.isdigit() for char in item):
        metal.append((item, similarity))
    # Step 4: Keep metal items that contain "alloy"
    elif "alloy" in item:
        metal.append((item, similarity))
    # Step 5: Keep property items that contain properties keywords    
    elif any(word in item for word in thermal_properties_words):
        thermal_property.append((item, similarity))
    elif any(word in item for word in mechanical_properties_words):
        mechanical_property.append((item, similarity))
    # Step 7: unfiltered items
    else:
        unfiltered_item.append((item, similarity))
    
print("\nThermal Properties:")
for item, sim in thermal_property:
    print(f"{item}: {sim}")

print("\nMechanical Properties:")
for item, sim in mechanical_property:
    print(f"{item}: {sim}")

print("\nProcess General:")
for item, sim in process_general:
    print(f"{item}: {sim}")

print("\nProcess Melt General:")
for item, sim in process_melt_general:
    print(f"{item}: {sim}")

print("\nProcess Melt Ded:")
for item, sim in process_melt_ded:
    print(f"{item}: {sim}")

print("\nProcess Melt PBF:")
for item, sim in process_melt_pbf:
    print(f"{item}: {sim}")

print("\nProcess Solid General:")
for item, sim in process_solid_general:
    print(f"{item}: {sim}")

print("\nProcess Solid Binder:")
for item, sim in process_solid_binder:
    print(f"{item}: {sim}")

print("\nProcess Solid Cold Spray:")
for item, sim in process_solid_cold_spray:
    print(f"{item}: {sim}")

print("\nProcess Solid Extrusion:")
for item, sim in process_solid_extrusion:
    print(f"{item}: {sim}")

print("\nProcess Solid Field Assisted:")
for item, sim in process_solid_field_assisted:
    print(f"{item}: {sim}")

print("\nProcess Solid Friction Based:")
for item, sim in process_solid_friction_based:
    print(f"{item}: {sim}")

print("\nMetal:")
for item, sim in metal:
    print(f"{item}: {sim}")

print("\nNon-Filtered Items:")
for item, sim in unfiltered_item:
    print(f"{item}: {sim}")
