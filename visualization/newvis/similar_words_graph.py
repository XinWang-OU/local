# ONLY ONE COLOR
import matplotlib.pyplot as plt
from collections import Counter
import pandas as pd


file_path = '/home/etica/Project/alloy2vec/visualization/newvis/gas_pipeline.xlsx'
df = pd.read_excel(file_path)

time_periods = df['Year Period'].unique().tolist()


distance_values = []
alloy_names = []

for period in time_periods:
    period_data = df[df['Year Period'] == period]
    alloy_names.append(period_data['Materials'].tolist())
    distance_values.append(period_data['Distance'].tolist())



# Preparing data for plotting
x_values, y_values, labels, label_sides, ha_values = [], [], [], [], []

for i, (period_names, period_values) in enumerate(zip(alloy_names, distance_values)):
    prev_value = None
    label_side = 'right'  # Default to right for the first time period

    for j, (name, value) in enumerate(zip(period_names, period_values)):
        if name == 'N/A':  # Skip 'N/A' labels
            continue
        x_values.append(i)
        y_values.append(value)
        labels.append(name)

        # Determine label side based on distance
        if prev_value is not None and abs(value - prev_value) < 0.005:
            label_side = 'left' if label_side == 'right' else 'right'
        ha_values.append(label_side)
        
        prev_value = value



# Adjusting figure size and plotting
plt.figure(figsize=(7, 5.83))  # Adjusted figsize for a more proportionate appearance
for x, y, label, ha in zip(x_values, y_values, labels, ha_values):
    plt.scatter(x, y, color='#00BEFCC3')
    plt.text(x, y, label, fontsize=16, ha=ha)

plt.gca().spines['top'].set_visible(False)
plt.gca().spines['right'].set_visible(False)
plt.xticks(range(len(time_periods)), time_periods, ) # rotation=45
plt.xlim(-1, len(time_periods))
plt.xlabel('Year', fontsize=16)
plt.ylabel('Cosine Distance', fontsize=16)
plt.ylim(0.29, 0.43)  # Adjusted y-axis range
# #nplt.title("KEYWORDS: high_entropy_alloy, multi-principal_element_alloy,\n"
#           "baseless_alloy, multi-component_alloy, complex_concentrated_alloy,\n"
#           "compositionally_complex_alloy", fontsize=14)
 # Added closing quotation mark
plt.tight_layout()
plt.show()


