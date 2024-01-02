#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 28 02:42:04 2023

@author: Talal
"""

import pandas as pd
import matplotlib.pyplot as plt


def create_dfs(path):
    df = pd.read_csv(path)
    df= df.set_index("Country Name")
    df = df.dropna()
    t_df = df.copy()
    t_df = t_df.drop(
        ['Indicator Name', 'Indicator Code', 'Country Code'], axis=1)

    t_df = t_df.T
    country_df = t_df.dropna()

    return df, country_df

def line_plot(df, title, xlabel, ylabel):
    countries = ['Chile', 'Zimbabwe',"India", "Tanzania","Netherlands", 
                 "Pakistan", "United States"]
    data = df.iloc[-10:]
    countries_df = data[countries]
    print(countries_df.head())
    plt.title(title)
    plt.plot(countries_df)
    plt.xlabel(xlabel)
    plt.xticks(rotation=90)
    plt.ylabel(ylabel)
    plt.legend(countries, bbox_to_anchor=(1.0, 1), loc="upper right")
    plt.show()
    
def create_bar_graph(data, title, xlabel, ylabel):
    countries = ['Chile', 'Zimbabwe',"India", "Tanzania", "Afghanistan",
                 "Liberia", "Netherlands", "Pakistan"]
    data = data.iloc[:10]
    data = data[countries]
    plt.figure(figsize=(8,6))
    # last_years_data = last_years_data.T
    ax = data.plot(kind='bar', figsize=(10, 6))
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    plt.legend(bbox_to_anchor=(1.0, 1), loc="upper right")
    plt.show()
    

path1 = "data/forestArea.csv"
path2 = "data/CO2.csv"
path3 = "data/electricityFromCoal.csv"

    
forest, forest_yearly = create_dfs(path1)
print(forest)
print(forest_yearly)
co2, co2_yearly = create_dfs(path2)
coal_elec, coal_elec_yearly = create_dfs(path3)

create_bar_graph(forest_yearly, "Forest Area", "Year", "Persons")
create_bar_graph(co2_yearly, "CO2 emissions (metric tons per capita)", 
                 "Year", "CO2")
line_plot(coal_elec_yearly, "Electricity production from coal sources (% of total)", 
                 "Year", "Electricity production from coal")
line_plot(co2_yearly, "CO2 emissions (metric tons per capita)", 
                 "Year", "CO2 Emissions")

print(coal_elec.index.tolist())




















