import re


pattern = re.compile(r'multi-?\s*principal\s+element\s+alloy', re.IGNORECASE)


file_path = '/home/etica/Project/alloy2vec/co_occurrence/extracted_2021_2019.txt'

try:
    count = 0

    with open(file_path, 'r') as file:
        for line in file:
            matches = pattern.findall(line)
            count += len(matches)

    print(f"Total occurrences found: {count}")
except Exception as e:
    print(f"An error occurred: {e}")
