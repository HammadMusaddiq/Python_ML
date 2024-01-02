import matplotlib.gridspec as gridspec
import matplotlib.colors as mcolors
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
from matplotlib.image import imread
from matplotlib import image as mpimg
import seaborn as sns
import pandas as pd
import numpy as np
import random
import textwrap

## Dataset
## https://www.kaggle.com/datasets/shariful07/student-flexibility-in-online-learning?select=students_adaptability_level_online_education.csv

data = pd.read_csv("students_adaptability_level_online_education.csv")

bins = [0, 5, 10, 15, 20, 25, 30, 40, 50, 60, 70, 80, 90]
labels = ['1-5', '5-10', '10-15', '15-20', '20-25', '25-30', '30-40', '40-50', '50-60', '60-70', '70-80', '80-90']

data['age_group'] = ''
for i in range(len(bins)-1):
    data.loc[(data['Age'] > bins[i]) & (data['Age'] <= bins[i+1]), 'age_group'] = labels[i]

def p_bar_plot(ax, x, y, x_axis_title, y_axis_title, title):
    # Define color scheme
    colors = ['#e74c3c', '#3498db', '#2ecc71', '#f1c40f', '#9b59b6', '#34495e', '#95a5a6', '#e67e22', '#1abc9c', '#7f8c8d']

    # Plot bars
    bars = ax.bar(x, y, color=colors)

    # Add labels to bars
    for bar in bars:
        height = bar.get_height()
        ax.annotate(f'{height:.0f}', xy=(bar.get_x() + bar.get_width() / 2, height), xytext=(0, 3),
                    textcoords="offset points", ha='center', va='bottom', fontsize=12)

    # Set axis labels and title
    ax.set_xlabel(x_axis_title, labelpad=10, color="red")
    ax.set_ylabel(y_axis_title, labelpad=10, color="red")
    ax.set_title(title, fontsize=12, fontweight='bold', pad=6, color ='white', loc='center')

    # Remove spines and set tick parameters
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.tick_params(axis='both', which='major', labelsize=12, length=8, width=1, pad=8)

    # Set axis limits and grid
    ax.set_ylim([0, max(y) + 100])
    ax.grid(axis='y', linestyle='--', alpha=0.7)
    # ax.grid(color ='grey', linestyle ='-.', linewidth = 0.5, alpha = 0.5)

    # Remove x-axis spines and set tick labels
    ax.spines['bottom'].set_linewidth(0.5)
    ax.spines['bottom'].set_color('white')  # set color of x-axis
    ax.spines['left'].set_color('white')   # set color of y-axis
    ax.tick_params(axis='x', which='major', labelsize=12, length=0, labelcolor = 'white')
    ax.tick_params(axis='y', labelcolor='white') # set color of y-axis values

    # using set_facecolor() method
    ax.set_facecolor("black")

    # Add legend
    ax.legend(bars, x, loc='upper right', fontsize=12, frameon=False, bbox_to_anchor=(1.2, 1), labelcolor = "white")


def chart_first(fig,ax1):

    #  fig color
    fig.set_facecolor("black")

    age = data['age_group'].value_counts().sort_index()
    p_bar_plot(ax1, age.index, age.values, 'Age Group', 'Count', 'Age Distribution')
    
    
    # # # Remove x, y Ticks
    # ax1.xaxis.set_ticks_position('none')
    # ax1.yaxis.set_ticks_position('none')
    
    # # Add padding between axes and labels
    # ax1.xaxis.set_tick_params(pad = 2)
    # ax1.yaxis.set_tick_params(pad = 5)

    # # Add x, y gridlines

    # ax1.set_xlabel("Institution Type",  labelpad=10, color="red")
    # ax1.set_ylabel("Frequency", labelpad=10, color="red")
    # ax1.set_title("Frequency Graph of Institution Type", fontsize = 12, fontweight ='bold', color ='white', loc='center', pad = 6)

    # ax1.tick_params(axis='x', rotation=0)


def chart_second(fig, ax2):
    ## Second Graph - Histogram + Line chart
    ax2.hist(data["Age"], density=True, color='green', alpha=0.6)

    # set custom start and end values for x-axis
    ax2.set_xlim(0, max(data['Age']) + 5)

    # compute kernel density estimate
    kde = sns.kdeplot(data['Age'], color='white', ax=ax2, linewidth=2)

    # get x and y values of the kde line
    x_kde, y_kde = kde.get_lines()[0].get_data()

    # plot a line from starting point of xlim to ending of xlim
    start, end = ax2.get_xlim()
    x_line = np.linspace(start, end, 1000)
    y_line = np.interp(x_line, x_kde, y_kde)
    ax2.plot(x_line, y_line, color='white', linewidth=2)

    # set axis text color to white
    ax2.xaxis.label.set_color('white')
    ax2.yaxis.label.set_color('white')
    ax2.tick_params(axis='x', colors='white')
    ax2.tick_params(axis='y', colors='white')

    # set grid properties
    ax2.grid(color='grey', linestyle='-.', linewidth=0.5, alpha=0.5)

    # set background color to black
    ax2.set_facecolor('black')

    ax2.set_xlabel("Age",  labelpad=10, color="red")
    ax2.set_ylabel("Frequency", labelpad=10, color="red")

    # set title
    ax2.set_title('Age Distribution', fontsize = 12, fontweight ='bold', color ='white', loc='center', pad = 6)


def pie_multichart():

    fig = plt.figure(figsize=(18, 6))

    gs = gridspec.GridSpec(ncols=1, nrows=1, figure=fig)
    # gs = fig.add_gridspec(nrows=1, ncols=1)
    ax1 = fig.add_subplot(gs[0])

    gs1 = gs[0].subgridspec(1, 3)
    ax1_1 = fig.add_subplot(gs1[0])
    ax1_2 = fig.add_subplot(gs1[1])
    ax1_3 = fig.add_subplot(gs1[2])

    # Define colors for each age group
    colors = {'5-10': '#FFC107', '10-15': '#FF5722', '15-20': '#2196F3', '20-25': '#4CAF50', '25-30': '#9C27B0'}
    wedgeprops = {'linewidth': 2, 'edgecolor': 'black'}

    # create pie charts for each flexibility level
    data[data['Flexibility Level'] == 'Low'].groupby(['age_group']).size().plot(kind='pie', ax=ax1_1, autopct='%1.1f%%', startangle=90, legend=False, textprops={'fontsize': 10}, colors=[colors[x] for x in data[data['Flexibility Level'] == 'Low']['age_group'].unique()], wedgeprops=wedgeprops)
    data[data['Flexibility Level'] == 'Moderate'].groupby(['age_group']).size().plot(kind='pie', ax=ax1_2, autopct='%1.1f%%', startangle=90, legend=False, textprops={'fontsize': 10}, colors=[colors[x] for x in data[data['Flexibility Level'] == 'Moderate']['age_group'].unique()], wedgeprops=wedgeprops)
    data[data['Flexibility Level'] == 'High'].groupby(['age_group']).size().plot(kind='pie', ax=ax1_3, autopct='%1.1f%%', startangle=90, legend=False, textprops={'fontsize': 10}, colors=[colors[x] for x in data[data['Flexibility Level'] == 'High']['age_group'].unique()], wedgeprops=wedgeprops)

    # set titles for subplots
    ax1_1.set_title('Low Flexibility', fontsize=14, fontweight='bold', color='white')
    ax1_2.set_title('Moderate Flexibility', fontsize=14, fontweight='bold', color='white')
    ax1_3.set_title('High Flexibility', fontsize=14, fontweight='bold', color='white')

    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)
    ax1.spines['left'].set_visible(False)
    ax1.spines['bottom'].set_visible(False)

    ax1.set_xticklabels([])
    ax1.set_yticklabels([])

    ax1.xaxis.set_ticks_position('none')
    ax1.yaxis.set_ticks_position('none')

    # set title for the main plot
    ax1.set_title('Flexibility Level by Age Group', fontsize=16, fontweight='bold', color = 'white')

    # Create list of patches for the legend
    legend_patches = [mpatches.Patch(color=color, label=age_group) for age_group, color in colors.items()]

    # Add legend to main plot
    ax1.legend(handles=legend_patches, loc='lower center', bbox_to_anchor=(0.5, 0), ncol=5, labelcolor="Black")

    #  fig color
    fig.set_facecolor("black")

    # Set the face color of the axis
    ax1.set_facecolor('black')

    plt.savefig('piechart_multi1.PNG', dpi=300, bbox_inches='tight')
    plt.close()


#calling funciton
pie_multichart()


def chart_third(fig,ax3,gs):

    img = mpimg.imread('piechart_multi1.PNG')

    # Show the image in the subplot, zoomed in to the center of the image
    height, width, channels = img.shape
    extent = [-0.5*width, 0.5*width, -0.5*height, 0.5*height]
    # ax3.imshow(img, extent=extent, aspect='auto')
    ax3.imshow(img)

    # Hide the axis ticks and labels
    ax3.set_xticks([])
    ax3.set_yticks([])

    #  fig color
    fig.set_facecolor("black")

    # Set the face color of the axis
    ax3.set_facecolor('black')

    # gs1 = gs.subgridspec(1, 3)
    # ax1_1 = fig.add_subplot(gs1[0])
    # ax1_2 = fig.add_subplot(gs1[1])
    # ax1_3 = fig.add_subplot(gs1[2])

    # # Define colors for each age group
    # colors = {'5-10': '#FFC107', '10-15': '#FF5722', '15-20': '#2196F3', '20-25': '#4CAF50', '25-30': '#9C27B0'}
    # wedgeprops = {'linewidth': 2, 'edgecolor': 'black'}

    # # create pie charts for each flexibility level
    # data[data['Flexibility Level'] == 'Low'].groupby(['age_group']).size().plot(kind='pie', ax=ax1_1, autopct='%1.1f%%', startangle=90, legend=False, textprops={'fontsize': 10}, colors=[colors[x] for x in data[data['Flexibility Level'] == 'Low']['age_group'].unique()], wedgeprops=wedgeprops)
    # data[data['Flexibility Level'] == 'Moderate'].groupby(['age_group']).size().plot(kind='pie', ax=ax1_2, autopct='%1.1f%%', startangle=90, legend=False, textprops={'fontsize': 10}, colors=[colors[x] for x in data[data['Flexibility Level'] == 'Moderate']['age_group'].unique()], wedgeprops=wedgeprops)
    # data[data['Flexibility Level'] == 'High'].groupby(['age_group']).size().plot(kind='pie', ax=ax1_3, autopct='%1.1f%%', startangle=90, legend=False, textprops={'fontsize': 10}, colors=[colors[x] for x in data[data['Flexibility Level'] == 'High']['age_group'].unique()], wedgeprops=wedgeprops)

    # # set titles for subplots
    # ax1_1.set_title('Low Flexibility', fontsize=14, fontweight='bold', color='white')
    # ax1_2.set_title('Moderate Flexibility', fontsize=14, fontweight='bold', color='white')
    # ax1_3.set_title('High Flexibility', fontsize=14, fontweight='bold', color='white')

    # ax3.spines['top'].set_visible(False)
    # ax3.spines['right'].set_visible(False)
    # ax3.spines['left'].set_visible(False)
    # ax3.spines['bottom'].set_visible(False)

    # ax3.set_xticklabels([])
    # ax3.set_yticklabels([])

    # ax3.xaxis.set_ticks_position('none')
    # ax3.yaxis.set_ticks_position('none')

    # # set title for the main plot
    # ax3.set_title('Flexibility Level by Age Group', fontsize=16, fontweight='bold', color = 'white')

    # # Create list of patches for the legend
    # legend_patches = [mpatches.Patch(color=color, label=age_group) for age_group, color in colors.items()]

    # # Add legend to main plot
    # ax3.legend(handles=legend_patches, loc='lower center', bbox_to_anchor=(0.5, 0), ncol=5, labelcolor="Black")

    # #  fig color
    # fig.set_facecolor("black")

    # # Set the face color of the axis
    # ax3.set_facecolor('black')


def chart_fourth(fig,ax4):
    ## fourth Graph -  Bar Chart
    # color = ["#"+''.join([random.choice('0123456789ABCDEF') for i in range(6)]) for j in range(len(data['Device'].unique()))]
    color = ['#ff9999','#66b3ff','#99ff99']
    data['Device'].value_counts().plot(ax=ax4, kind='barh', color = color , width = 0.5)

    # Remove axes splines
    # for s in ['top', 'bottom', 'left', 'right']:
    for s in ['top', 'right']:
        ax4.spines[s].set_visible(False)
    
    # # Remove x, y Ticks
    ax4.xaxis.set_ticks_position('none')
    ax4.yaxis.set_ticks_position('none')
    
    # Add padding between axes and labels
    ax4.xaxis.set_tick_params(pad = 2)
    ax4.yaxis.set_tick_params(pad = 5)

    #  fig color
    fig.set_facecolor("black")

    # using set_facecolor() method
    ax4.set_facecolor("black")

    ax4.spines['bottom'].set_color('blue')  # set color of x-axis
    ax4.spines['left'].set_color('blue')   # set color of y-axis

    ax4.tick_params(axis='x', labelcolor='white')  # set color of x-axis values
    ax4.tick_params(axis='y', labelcolor='white') # set color of y-axis values

    # Add x, y gridlines
    ax4.grid(color ='grey', linestyle ='-.', linewidth = 0.5, alpha = 0.5)

    # Add annotation to bars
    for i, c in zip(ax4.patches, color):
        ax4.text(i.get_width()+0.3, i.get_y()+0.2, str(round((i.get_width()), 2)), fontsize = 10, fontweight ='bold', color = c)

    ax4.set_xlabel("Device",  labelpad=10, color="red")
    ax4.set_ylabel("Frequency", labelpad=10, color="red") 
    ax4.set_title("Frequency Graph of Device", fontsize = 12, fontweight ='bold', color ='white', loc='center', pad = 6)

    for container in ax4.containers:
        for i, bar in enumerate(container):
            bar.set_color(mcolors.to_rgba(color[i]))


def chart_fifth(fig,ax5):
    color = ['#ff9999','#66b3ff','#99ff99']
    data.groupby(['age_group', 'Flexibility Level']).size().unstack().plot(kind='bar', ax=ax5, color=color)
    # Add x, y gridlines
    ax5.grid(color ='grey', linestyle ='-.', linewidth = 0.5, alpha = 0.5)

    for s in ['top', 'right']:
        ax5.spines[s].set_visible(False)

    # # Remove x, y Ticks
    ax5.xaxis.set_ticks_position('none')
    ax5.yaxis.set_ticks_position('none')
    
    # Add padding between axes and labels
    ax5.xaxis.set_tick_params(pad = 2)
    ax5.yaxis.set_tick_params(pad = 5)

    #  fig color
    fig.set_facecolor("black")

    # using set_facecolor() method
    ax5.set_facecolor("black")

    ax5.spines['bottom'].set_color('blue')  # set color of x-axis
    ax5.spines['left'].set_color('blue')   # set color of y-axis

    ax5.tick_params(axis='x', labelcolor='white')  # set color of x-axis values
    ax5.tick_params(axis='y', labelcolor='white') #

    ax5.set_xlabel('Age Group',  labelpad=10, color="red")
    ax5.set_ylabel('Count',  labelpad=10, color="red")
    ax5.set_title('Flexibility Level by Age Group', fontsize = 12, fontweight ='bold', color ='white', loc='center', pad = 6)
    
    ax5.legend(labelcolor='white')


def create_figure():
    with plt.style.context(("seaborn","ggplot")):
        fig = plt.figure(constrained_layout=True, figsize=(12,14))

        specs = gridspec.GridSpec(ncols=2, nrows=3, figure=fig) ## Declaring 3x2 figure.
        specs.update(hspace=0.2, wspace=0.01)

        ax1 = fig.add_subplot(specs[0, 0]) ## First Row First Column
        ax2 = fig.add_subplot(specs[0, 1]) ## First Row Second Colums
        ax3 = fig.add_subplot(specs[1, :]) ## Second Row first column
        ax4 = fig.add_subplot(specs[2, 0]) ## Thirt Row 
        ax5 = fig.add_subplot(specs[2, 1]) ## Thirt Row 

        
        chart_first(fig,ax1)
        chart_second(fig,ax2)
        chart_third(fig,ax3,specs[1, :])
        chart_fourth(fig,ax4)
        chart_fifth(fig,ax5)

        ax3.set_position([0.0, 0.0, 0.9, 1])
        
        text = 'I made a Visualization Dashboard, using a kaggle dataset. Multiple charts have been plotted from different types (Bar Chart,\
            Pie Chart, Line Chart, and Text Chat).'
        wrapped_text = textwrap.wrap(text, width=12)

        fig.text(0.92, 0.4, '\n'.join(wrapped_text), fontsize=12, bbox=dict(boxstyle='round', facecolor='none', edgecolor='grey', linewidth=2, pad=1, alpha=0.5), color='white')

        fig.suptitle("Student Adaptability Level in Online Education (FullName+StudentID)", fontsize=20, fontweight='bold', color='white')
        plt.savefig('StudentID.PNG', dpi=300, bbox_inches='tight')
        plt.close(fig)
        return fig
    

create_figure()

import os
os.remove("piechart_multi1.PNG")