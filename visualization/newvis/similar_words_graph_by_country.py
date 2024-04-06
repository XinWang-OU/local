import matplotlib.pyplot as plt
import pandas as pd

# 加载数据从Excel文件
file_path = '/home/etica/Project/alloy2vec/visualization/newvis/gas_pipeline.xlsx' 
df = pd.read_excel(file_path)

# 定义默认颜色
default_color = '#5EC286'

# 检查'Country'列是否存在
country_column_exists = 'Country' in df.columns

if country_column_exists:
    country_colors = {
        'US': '#1f77b4',  # 蓝色
        'Germany': '#ff7f0e',  # 橙色
        'Japan': '#2ca02c',  # 绿色
        'China': '#d62728',  # 红色
        'UK': '#9467bd',  # 紫色
        'Russia/USSR': '#e377c2',  # 粉色
        'US and Europe': '#7f7f7f',  # 灰色
        'Europe': '#bcbd22',  # 黄绿色
        'MPEA': '#8C5119',  # 褐色
        'non-MPEA' : '#1f77b4'  # 蓝色
    }
else:
    country_colors = {}

# 初始化一个集合来追踪图上实际出现的国家
appeared_countries = set()

# 准备绘图数据，使用自定义国家颜色或默认颜色
x_values, y_values, labels, label_sides, ha_values, point_colors = [], [], [], [], [], []

for i, period in enumerate(df['Year Period'].unique()):
    period_data = df[df['Year Period'] == period]
    prev_value = None
    label_side = 'right'

    for _, row in period_data.iterrows():
        if row['Materials'] == 'N/A':
            continue
        x_values.append(i)
        y_values.append(row['Distance'])
        labels.append(row['Materials'])
        if country_column_exists:
            appeared_countries.add(row['Country'])
            point_colors.append(country_colors.get(row['Country'], 'grey'))  # 如果未指定，默认为'grey'
        else:
            point_colors.append(default_color)

        # 特别为1989-2003年度的点调整标签位置
        if row['Year Period'] == '1989-2003' and abs(row['Distance'] - 0.560245513916016) < 0.0001:
            label_side = 'left'
        else:
            # 根据距离确定标签侧面
            if prev_value is not None and abs(row['Distance'] - prev_value) < 0.01:
                label_side = 'left' if label_side == 'right' else 'right'
        ha_values.append(label_side)
        
        prev_value = row['Distance']


# 调整图形大小并使用自定义国家颜色或默认颜色进行绘图
plt.figure(figsize=(10, 6))
for x, y, label, ha, color in zip(x_values, y_values, labels, ha_values, point_colors):
    plt.scatter(x, y, color=color)
    plt.text(x, y, label, fontsize=16, ha=ha)

# 移除顶部和右侧轮廓线
plt.gca().spines['top'].set_visible(False)
plt.gca().spines['right'].set_visible(False)

# 设置x轴刻度和限制
plt.xticks(range(len(df['Year Period'].unique().tolist())), df['Year Period'].unique().tolist()) #rotation=45
plt.xlim(-1, len(df['Year Period'].unique().tolist()))
plt.xlabel('Year', fontsize=16)
plt.ylabel('Cosine Distance', fontsize=16)
plt.ylim(0.465,0.635)


if country_column_exists:
    legend_handles = [plt.Line2D([0], [0], marker='o', color='w', label=country, 
                                 markersize=14, markerfacecolor=color) for country, color in country_colors.items() if country in appeared_countries]
    plt.legend(handles=legend_handles, bbox_to_anchor=(1, 0), loc='lower left', fontsize=16) # title='Country',
else:
    plt.scatter([], [], color=default_color, label='Default')
    plt.legend(title='Color Legend', bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=16)

plt.tick_params(axis='x', labelsize=16)  
plt.tick_params(axis='y', labelsize=16)  

plt.tight_layout()
plt.show()
