import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

#url = "https://data.worldbank.org/topic/climate-change?end=2019&start=2010"
#data_url = "https://api.worldbank.org/v2/en/topic/19?downloadformat=csv"

data_path_main = "API_19_DS2_en_csv_v2_5455435\API_19_DS2_en_csv_v2_5455435.csv"
indicators = ["Urban population", "Population, total", "Population growth (annual %)"]

def process_data(filename):
    df_data = pd.read_csv(filename, header=2)
    df = df_data.drop('Unnamed: 67', axis=1)
    df = df.fillna(df.mean())
    data_years = df.drop(['Country Name',  'Country Code', 'Indicator Name', 'Indicator Code'], axis=1).copy()
    data_countries = df.set_index("Country Name").copy()
    return data_years, data_countries

def select_rand_data(df):# Select 10 random countries from the data
    countries = np.random.choice(df.index.unique(), size=10, replace=False)
    data_ind1 = df.loc[df.index.isin(countries) & (df['Indicator Name'] == indicators[0])]
    data_ind1 = data_ind1[data_ind1.columns[-13:-3]] #select data from year 2010-2019
    data_ind2 = df.loc[df.index.isin(countries) & (df['Indicator Name'] == indicators[1])]
    data_ind2 = data_ind2[data_ind2.columns[-13:-3]] #select data from year 2010-2019 
    data_ind3 = df.loc[df.index.isin(countries) & (df['Indicator Name'] == indicators[2])]
    data_ind3 = data_ind3[data_ind3.columns[-23:-3]] #select data from year 2010-2019 
    
    return data_ind1, data_ind2, data_ind3


def plot_bar(data, indicator):
    data_avg = data.mean()
    plt.figure()
    plt.bar(data.index, data_avg)
    plt.title(indicator)
    plt.xlabel('Country')
    plt.ylabel(indicator + ' average from 2010 to 2019')
    plt.xticks(rotation=90)
    plt.ticklabel_format(axis='y', style='plain')
    plt.savefig(indicator+".png", bbox_inches='tight')
    plt.show()

def bar_graph(data, indicator):
    ax = data.plot(kind='bar', figsize=(10, 6))
    ax.set_title(indicator)
    ax.set_xlabel("Country")
    ax.set_ylabel(indicator + " Year Wise")
    ax.set_xticklabels(data.index, rotation=90)
    ax.ticklabel_format(axis='y', style='plain')
    plt.legend()
    plt.savefig(indicator+"bar.png", bbox_inches='tight')
    plt.show()


def plot_line(data, indicator):
    plt.figure(figsize=(12,6))
    for i in range(len(data.index)):
        plt.plot(data.columns, data.iloc[i])
    plt.title(indicator)
    plt.xlabel('Year')
    plt.ylabel(indicator)
    plt.legend(data.index, loc='upper left')
    plt.savefig(indicator+"line.png", bbox_inches='tight')
    plt.show()


data_years, data_countries = process_data(data_path_main)
data_ind1, data_ind2, data_ind3 = select_rand_data(data_countries)

plot_bar(data_ind1,indicators[0])
plot_bar(data_ind2,indicators[1])

bar_graph(data_ind1, indicators[0])
bar_graph(data_ind2, indicators[1])

plot_line(data_ind3, indicators[2])

print("Data Exploration for Urban population Indicator")
print(data_ind1.describe())
print("Median:", data_ind1.median())
print("Variance:", data_ind1.var())

print("Data Exploration for Total population Indicator")
print(data_ind1.describe())
print("Median:", data_ind2.median())
print("Variance:", data_ind2.var())
