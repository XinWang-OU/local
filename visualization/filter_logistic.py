import numpy as np
from gensim import matutils

def predict_combined_context(w2v_model, words, topn=500):
    vecs = [w2v_model.wv[word] for word in words if word in w2v_model.wv]
    if len(vecs) == 0:
        return []

    combined_vec = np.mean(vecs, axis=0)
    prob_values = np.exp(np.dot(combined_vec, w2v_model.trainables.syn1neg.T))
    prob_values /= np.sum(prob_values)
    top_indices = matutils.argsort(prob_values, topn=topn, reverse=True)
    return [(w2v_model.wv.index2word[index], prob_values[index]) for index in top_indices]

#words_to_predict = ["high", "porosity"]
#top_context_words = predict_combined_context(w2v_model, words_to_predict, topn=500)
#top_context_words_string = '\n'.join(f"{word}: {similarity}" for word, similarity in top_context_words)

def read_keywords(file_paths):
    words_set = set()
    for file_path in file_paths:
        with open(file_path, 'r') as file:
            for line in file:
                word = line.strip()  # 移除每行首尾的空白字符
                words_set.add(word)  # 将单个单词加入集合
    return list(words_set)  # 将集合转换为列表并返回

first_alloy_files = ['/home/etica/Project/alloy2vec/keywords/new_keyword/first_words/AA_family_all.txt',
                      '/home/etica/Project/alloy2vec/keywords/new_keyword/first_words/alloy.txt']
first_thermal_properties_file = ['/home/etica/Project/alloy2vec/keywords/new_keyword/first_words/thermal_properties.txt']
first_mechanical_properties_file = ['/home/etica/Project/alloy2vec/keywords/new_keyword/first_words/mechanical_properties.txt']
first_process_general_file = ['/home/etica/Project/alloy2vec/keywords/new_keyword/first_words/process_general.txt']
first_process_melt_ded_file = ['/home/etica/Project/alloy2vec/keywords/new_keyword/first_words/process_melt_ded.txt']
first_process_melt_general_file = ['/home/etica/Project/alloy2vec/keywords/new_keyword/first_words/process_melt_general.txt']
first_process_melt_pbf_file = ['/home/etica/Project/alloy2vec/keywords/new_keyword/first_words/process_melt_pbf.txt']
first_process_solid_binder_file = ['/home/etica/Project/alloy2vec/keywords/new_keyword/first_words/process_solid_binder.txt']
first_process_solid_cold_spray_file = ['/home/etica/Project/alloy2vec/keywords/new_keyword/first_words/process_solid_cold_spray.txt']
first_process_solid_extrusion_file = ['/home/etica/Project/alloy2vec/keywords/new_keyword/first_words/process_solid_extrusion.txt']
first_process_solid_field_assisted_file = ['/home/etica/Project/alloy2vec/keywords/new_keyword/first_words/process_solid_field_assisted.txt']
first_process_solid_friction_based_file = ['/home/etica/Project/alloy2vec/keywords/new_keyword/first_words/process_solid_friction_based.txt']
first_process_solid_general_file = ['/home/etica/Project/alloy2vec/keywords/new_keyword/first_words/process_solid_general.txt']

first_alloy_words = read_keywords(first_alloy_files)
first_thermal_properties_words = read_keywords(first_thermal_properties_file)
first_mechanical_properties_words = read_keywords(first_mechanical_properties_file)
first_process_general_words = read_keywords(first_process_general_file)
first_process_melt_ded_words = read_keywords(first_process_melt_ded_file)
first_process_melt_general_words = read_keywords(first_process_melt_general_file)
first_process_melt_pbf_words = read_keywords(first_process_melt_pbf_file)
first_process_solid_binder_words = read_keywords(first_process_solid_binder_file)
first_process_solid_cold_spray_words = read_keywords(first_process_solid_cold_spray_file)
first_process_solid_extrusion_words = read_keywords(first_process_solid_extrusion_file)
first_process_solid_field_assisted_words = read_keywords(first_process_solid_field_assisted_file)
first_process_solid_friction_based_words = read_keywords(first_process_solid_friction_based_file)
first_process_solid_general_words = read_keywords(first_process_solid_general_file)


other_alloy_files = ['/home/etica/Project/alloy2vec/keywords/new_keyword/other_words/similar_AA_family_all.txt',
                        '/home/etica/Project/alloy2vec/keywords/new_keyword/other_words/similar_alloy.txt']
other_thermal_properties_file = ['/home/etica/Project/alloy2vec/keywords/new_keyword/other_words/similar_thermal_properties.txt']
other_mechanical_properties_file = ['/home/etica/Project/alloy2vec/keywords/new_keyword/other_words/similar_mechanical_properties.txt']
other_process_general_file = ['/home/etica/Project/alloy2vec/keywords/new_keyword/other_words/similar_process_general.txt']
other_process_melt_ded_file = ['/home/etica/Project/alloy2vec/keywords/new_keyword/other_words/similar_process_melt_ded.txt']
other_process_melt_general_file = ['/home/etica/Project/alloy2vec/keywords/new_keyword/other_words/similar_process_melt_general.txt']
other_process_melt_pbf_file = ['/home/etica/Project/alloy2vec/keywords/new_keyword/other_words/similar_process_melt_pbf.txt']
other_process_solid_binder_file = ['/home/etica/Project/alloy2vec/keywords/new_keyword/other_words/similar_process_solid_binder.txt']
other_process_solid_cold_spray_file = ['/home/etica/Project/alloy2vec/keywords/new_keyword/other_words/similar_process_solid_cold_spray.txt']
other_process_solid_extrusion_file = ['/home/etica/Project/alloy2vec/keywords/new_keyword/other_words/similar_process_solid_extrusion.txt']
other_process_solid_field_assisted_file = ['/home/etica/Project/alloy2vec/keywords/new_keyword/other_words/similar_process_solid_field_assisted.txt']
other_process_solid_friction_based_file = ['/home/etica/Project/alloy2vec/keywords/new_keyword/other_words/similar_process_solid_friction_based.txt']
other_process_solid_general_file = ['/home/etica/Project/alloy2vec/keywords/new_keyword/other_words/similar_process_solid_general.txt']

other_alloy_words = read_keywords(other_alloy_files)
other_thermal_properties_words = read_keywords(other_thermal_properties_file)
other_mechanical_properties_words = read_keywords(other_mechanical_properties_file)
other_process_general_words = read_keywords(other_process_general_file)
other_process_melt_ded_words = read_keywords(other_process_melt_ded_file)
other_process_melt_general_words = read_keywords(other_process_melt_general_file)
other_process_melt_pbf_words = read_keywords(other_process_melt_pbf_file)
other_process_solid_binder_words = read_keywords(other_process_solid_binder_file)
other_process_solid_cold_spray_words = read_keywords(other_process_solid_cold_spray_file)
other_process_solid_extrusion_words = read_keywords(other_process_solid_extrusion_file)
other_process_solid_field_assisted_words = read_keywords(other_process_solid_field_assisted_file)
other_process_solid_friction_based_words = read_keywords(other_process_solid_friction_based_file)
other_process_solid_general_words = read_keywords(other_process_solid_general_file)


ELEMENTS = ["Li", "Be", "Na", "Mg", "Al", "K",
            "Ca", "Sc", "Ti", "V", "Cr", "Mn", "Fe", "Co", "Ni", "Cu", "Zn", "Ga",
            "Rb", "Sr", "Y", "Zr", "Nb", "Mo", "Tc", "Ru", "Rh", "Pd", "Ag", "Cd", "In", "Sn", "Sb",
            "Cs", "Ba", "La", "Ce", "Pr", "Nd", "Pm", "Sm", "Eu", "Gd", "Tb", "Dy", "Ho", "Er", "Tm", "Yb",
            "Lu", "Hf", "Ta", "W", "Re", "Os", "Ir", "Pt", "Au", "Hg", "Tl", "Pb", "Bi", "Po",
            "Ra", "Ac", "Th", "Pa", "U", "Np", "Pu", "Am", "Cm", "Bk", "Cf", "Es"]

ELEMENT_NAMES = ["Lithium", "Beryllium", "Sodium", "Magnesium", "Aluminium", "Potassium",
"Calcium", "Scandium", "Titanium", "Vanadium", "Chromium", "Manganese", "Iron", "Cobalt", "Nickel", "Copper", "Zinc", "Gallium",
"Rubidium", "Strontium", "Yttrium", "Zirconium", "Niobium", "Molybdenum", "Technetium", "Ruthenium", "Rhodium", "Palladium", "Silver", "Cadmium", "Indium", "Tin", "Antimony",
"Cesium", "Barium", "Lanthanum", "Cerium", "Praseodymium", "Neodymium", "Promethium", "Samarium", "Europium", "Gadolinium", "Terbium", "Dysprosium", "Holmium", "Erbium", "Thulium", "Ytterbium",
"Lutetium", "Hafnium", "Tantalum", "Tungsten", "Rhenium", "Osmium", "Iridium", "Platinum", "Gold", "Mercury", "Thallium", "Lead", "Bismuth", "Polonium",
"Radium", "Actinium", "Thorium", "Protactinium", "Uranium", "Neptunium", "Plutonium", "Americium", "Curium", "Berkelium", "Californium", "Einsteinium"]

def filter_words(top_context_words):
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
        #Below filter first words
        if any(element in item for element in ELEMENTS):
            metal.append((item, similarity))
        elif any(element in item for element in ELEMENT_NAMES):
            metal.append((item, similarity))
        # Step 3: Keep metal items that contain any word from alloy words (full match)
        elif item in first_alloy_words:
            metal.append((item, similarity))
        # Step 3: Keep metal items that contain both letters and digits （是否需要？）
        #elif any(char.isalpha() for char in item) and any(char.isdigit() for char in item):
            #metal.append((item, similarity))
        # Step 4: Keep metal items that contain "alloy"
        elif "alloy" in item:
            metal.append((item, similarity))
        elif "steel" in item:
            metal.append((item, similarity))
        elif any(word in item for word in first_process_general_words):
            process_general.append((item, similarity))
        elif any(word in item for word in first_process_melt_ded_words):
            process_melt_ded.append((item, similarity))
        elif any(word in item for word in first_process_melt_general_words):
            process_melt_general.append((item, similarity))
        elif any(word in item for word in first_process_melt_pbf_words):
            process_melt_pbf.append((item, similarity))
        # elif any(word in item for word in process_solid_binder_words):
        #     process_solid_binder.append((item, similarity))
        elif item in first_process_solid_binder_words:
            process_solid_binder.append((item, similarity))
        elif any(word in item for word in first_process_solid_cold_spray_words):
            process_solid_cold_spray.append((item, similarity))
        elif any(word in item for word in first_process_solid_extrusion_words):
            process_solid_extrusion.append((item, similarity))
        elif any(word in item for word in first_process_solid_field_assisted_words):
            process_solid_field_assisted.append((item, similarity))
        elif any(word in item for word in first_process_solid_friction_based_words):
            process_solid_friction_based.append((item, similarity))
        elif any(word in item for word in first_process_solid_general_words):
            process_solid_general.append((item, similarity))
    
        # Step 5: Keep property items that contain properties keywords    
        elif item in first_thermal_properties_words:
            thermal_property.append((item, similarity))
        elif item in first_mechanical_properties_words:
            mechanical_property.append((item, similarity))

        #Below filter other words
        elif item in other_alloy_words:
            metal.append((item, similarity))
        elif item in other_process_general_words:
            process_general.append((item, similarity))
        elif item in other_process_melt_ded_words:
            process_melt_ded.append((item, similarity))
        elif item in other_process_melt_general_words:
            process_melt_general.append((item, similarity))
        elif  item in other_process_melt_pbf_words:
            process_melt_pbf.append((item, similarity))
        elif  item in other_process_solid_cold_spray_words:
            process_solid_cold_spray.append((item, similarity))
        elif  item in other_process_solid_extrusion_words:
            process_solid_extrusion.append((item, similarity))
        elif item in other_process_solid_field_assisted_words:
            process_solid_field_assisted.append((item, similarity))
        elif  item in other_process_solid_friction_based_words:
            process_solid_friction_based.append((item, similarity))
        elif item in other_process_solid_general_words:
            process_solid_general.append((item, similarity))
        elif item in other_mechanical_properties_words:
            mechanical_property.append((item, similarity))
        elif item in other_thermal_properties_words:
            thermal_property.append((item, similarity))          
        # Step 7: unfiltered items
        else:
            unfiltered_item.append((item, similarity))
        
        return {
            "metal": metal,
            "thermal_property": thermal_property,
            "mechanical_property": mechanical_property,
            "process_general": process_general,
            "process_melt_ded": process_melt_ded,
            "process_melt_general": process_melt_general,
            "process_melt_pbf": process_melt_pbf,
            "process_solid_binder": process_solid_binder,
            "process_solid_cold_spray": process_solid_cold_spray,
            "process_solid_extrusion": process_solid_extrusion,
            "process_solid_field_assisted": process_solid_field_assisted,
            "process_solid_friction_based": process_solid_friction_based,
            "process_solid_general": process_solid_general,
            "unfiltered_item": unfiltered_item
        }