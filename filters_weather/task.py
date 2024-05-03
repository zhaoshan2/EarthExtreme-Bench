import numpy as np
import matplotlib.pyplot as plt
# Display task difficulties
data = [0.6,0.5,0.9,0.1,0, 0.3 ]
# Designed for heatwave
categories = ['spatial resolution', 'spatial coverage', 'forecasting horizon','task transfering', 'data modality', 'temporal resolution']
#levels
#Take ClimaX for example:
# spatial resolution: 5.6 -> 0.25 +1
# spatial coverage: global -> regional +1
# forecast horizons: 7 days -> 28 days +1
# task tranfering: weather-> weather, t2m -> t2m
# data modality: era5 -> era5
# temporal resolution:  hourly -> daily +1
# -> task difficulty: 4

#Take Phitiv for example:
# spatial resolution: 30m -> 0.25 +1
# spatial coverage: local1 -> local2 +1
# forecast horizons: 1 day -> 28 days +1
# task tranfering: EO -> Weather +1
# data modality: Satellite -> era5 +1
# temporal resolution:  5-minute -> daily +1
# -> task difficulty: 6

for index, row in data.iterrows():
    de1_labels = row.index[1:]
    name = row[0]
    # Radar (Spider) Chart
    categories = list(de1_labels)
    N = len(categories)
    values = row[1:].tolist()
    values += values[:1]  # repeat the first value to close the circular graph
    angles = [n / float(N) * 2 * 3.14159 for n in range(N)]
    angles += angles[:1]
with plt.style.context(['science']):
    plt.figure(figsize=(8, 8))
    ax = plt.subplot(111, polar=True)
    ax.plot(angles, values)
    ax.fill(angles, values, 'teal', alpha=0.1)
    # Set the x-ticks (category labels)
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(categories, fontsize=28)
    custom_y_ticks = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.64]
    ax.set_yticks(custom_y_ticks)
    tolerance = 1e-6
    y_tick_labels = [f'{tick:.1f}' if abs(tick - round(tick / 0.2) * 0.2) < tolerance else '' for tick in
                     custom_y_ticks]
    ax.set_yticklabels(y_tick_labels)
    # 调整刻度值的字体大小
    ax.tick_params(axis='y', labelsize=28)  # y轴刻度值
    # Rotate each label to the specified angle
    for label, angle in zip(ax.get_xticklabels(), angles):
        if angle in [0, 3.14159]:
            label.set_horizontalalignment('left')
        elif 0 < angle < 1.6 or angle > 4.7:
            label.set_horizontalalignment('left')
        else:
            label.set_horizontalalignment('right')
    plt.tight_layout()
    plt.savefig(name+".png", dpi=150)
    plt.show()