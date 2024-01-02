import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
import pandas as pd
import textwrap
import random

# https://www.kaggle.com/datasets/muthuj7/weather-dataset?resource=download&select=weatherHistory.csv
dataset = pd.read_csv("weatherHistory.csv")

# Extract random number of records from dataset
ext_num_records = 20000
dataset = dataset.sample(n=ext_num_records, random_state=random.seed())
dataset.reset_index(drop=True, inplace=True)

# Converting to DataTime Format
dataset["Formatted Date"] = pd.to_datetime(dataset["Formatted Date"], utc=True)

# Extracting Information from Dataset
data = dataset[['Apparent Temperature (C)','Humidity','Formatted Date']].copy()
data = data.set_index('Formatted Date')
data = data.resample('M').mean()
data_of_april = data[data.index.month==4]

# Creating a 3x2 grid of subplots
fig = plt.figure(figsize=(25, 20))
gs = gridspec.GridSpec(3,2, width_ratios=[2, 1])

# Adding the first plot to the first row and first column
ax1 = fig.add_subplot(gs[0, 0])
ax1.set_xticklabels(ax1.get_xticklabels(), rotation=-30)
sns.barplot(data=dataset, x="Summary", y="Temperature (C)",hue="Precip Type", ax=ax1)
ax1.set_title("Average Temperature by Weather Summary and Precipitation Type")

# Adding the second plot to the first row and second column
ax2 = fig.add_subplot(gs[0, 1])
sns.barplot(data=data_of_april, x='Apparent Temperature (C)', y='Humidity', ax=ax2)
ax2.set_xticklabels(ax2.get_xticklabels(), rotation=-30)
ax2.set_title('Relation between temperature and humidity for the month of April')

# Adding the third plot to the second row and first columns
ax3 = fig.add_subplot(gs[1,0])
ax3.plot(data_of_april, marker='o',label=['Apparent Temperature (C)','Humidity'])
ax3.legend(title=['Apparent Temperature (C)','Humidity'], loc = 'center right',fontsize = 10)
ax3.set_title('Relation between temperature and humidity for the month of April')

# Adding the fourth plot to the second row and second columns
ax4 = fig.add_subplot(gs[1,1])
sns.heatmap(dataset.corr('spearman').abs()[['Temperature (C)']].sort_values('Temperature (C)'), ax=ax4)
ax4.set_title("Correlation of Temperature (C) with Other Variables")

# Adding the fifth plot to the third and last row and both columns
ax5 = fig.add_subplot(gs[2:, :])
sns.histplot(data=dataset, x="Temperature (C)", hue="Summary", multiple="stack", ax=ax5)
ax5.set_title("Analysis of Weather Conditions with Temperature(C)")

# Adjusting the spacing between subplots
plt.subplots_adjust(hspace=0.4)

# Adding text to the plot
text = 'I made a Visualization Dashboard, using a kaggle dataset. 5 charts have been plotted from different types (Bar Chart,\
    Histogram, Heatmap, and Line Chart).'

wrapped_text = textwrap.wrap(text, width=58)

fig.text(0.7, 0.27, '\n'.join(wrapped_text), fontsize=12, bbox=dict(boxstyle='round', facecolor='none', edgecolor='grey', linewidth=2, pad=1, alpha=0.5), color='Black')

fig.suptitle("Weather Data Analysis (FullName+StudentID)", fontsize=18, fontweight='bold', color='Black')
plt.savefig('StudentID-2.PNG', dpi=300, bbox_inches='tight')
plt.close(fig)
