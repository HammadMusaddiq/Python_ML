import matplotlib.gridspec as gridspec
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import random
import textwrap

## Dataset
## https://www.kaggle.com/datasets/thedevastator/higher-education-predictors-of-student-retention?select=dataset.csv

data = pd.read_csv("students_adaptability_level_online_education.csv")

def chart_first(fig,ax1):
    ## First Graph - Bar Chart
    data['Institution Type'].value_counts().plot(ax=ax1, kind='bar', color ='royalblue', width = 0.3)
    # for s in ['top', 'bottom', 'left', 'right']:
    for s in ['top', 'right']:
        ax1.spines[s].set_visible(False)
    
    # # Remove x, y Ticks
    ax1.xaxis.set_ticks_position('none')
    ax1.yaxis.set_ticks_position('none')
    
    # Add padding between axes and labels
    ax1.xaxis.set_tick_params(pad = 2)
    ax1.yaxis.set_tick_params(pad = 5)

    #  fig color
    fig.set_facecolor("black")

    # using set_facecolor() method
    ax1.set_facecolor("black")

    ax1.spines['bottom'].set_color('white')  # set color of x-axis
    ax1.spines['left'].set_color('white')   # set color of y-axis

    ax1.tick_params(axis='x', labelcolor='white')  # set color of x-axis values
    ax1.tick_params(axis='y', labelcolor='white') # set color of y-axis values

    # Add x, y gridlines
    ax1.grid(color ='grey', linestyle ='-.', linewidth = 0.5, alpha = 0.5)

    ax1.set_xlabel("Institution Type",  labelpad=10, color="red")
    ax1.set_ylabel("Frequency", labelpad=10, color="red")
    ax1.set_title("Frequency Graph of Institution Type", fontsize = 12, fontweight ='bold', color ='white', loc='center', pad = 6)

    ax1.tick_params(axis='x', rotation=0)


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



def chart_third(fig,ax3):
    ## Third Graph - Histogram
    color = {"Moderate" : "tab:red", "Low" : "tab:green", "High" : "tab:blue"}
    
    for level in data["Flexibility Level"].unique():
        ax3.scatter(data[data["Flexibility Level"]==level]["Age"],
                    data[data["Flexibility Level"]==level]["Education Level"],
                    c=color[level],
                    s=200,
                    alpha=0.6,
                    label=level)
    #  fig color
    fig.set_facecolor("black")

    # Set the face color of the axis
    ax3.set_facecolor('black')

    # set axis text color to light gray
    ax3.tick_params(axis='x', colors='white')
    ax3.tick_params(axis='y', colors='white')

    ax3.set_xlabel("Age", color="red")
    ax3.set_ylabel("Education Level", color="red")
    ax3.set_title("Education Level vs Age Scatter Plot", color="white")

    # set the color of the legend title
    legend = ax3.legend(title="Flexibility Level", loc="center left", title_fontsize=12, labelcolor='white', bbox_to_anchor=(1, 0.5))
    legend.get_title().set_color('white')


def chart_fourth(fig,ax4):
    ## fourth Graph -  Bar Chart
    color = ["#"+''.join([random.choice('0123456789ABCDEF') for i in range(6)]) for j in range(len(data['Device'].unique()))]
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


def create_figure():
    with plt.style.context(("seaborn","ggplot")):
        fig = plt.figure(constrained_layout=True, figsize=(10,12))

        specs = gridspec.GridSpec(ncols=2, nrows=3, figure=fig, width_ratios=[1, 2]) ## Declaring 3x2 figure.
        specs.update(wspace=0.5, hspace=1)

        ax1 = fig.add_subplot(specs[0, 0]) ## First Row First Column
        ax2 = fig.add_subplot(specs[0, 1]) ## First Row Second Colums
        ax3 = fig.add_subplot(specs[1, :]) ## Second Row 
        ax4 = fig.add_subplot(specs[2, :]) ## Thirt Row 

        
        chart_first(fig,ax1)
        chart_second(fig,ax2)
        chart_third(fig,ax3)
        chart_fourth(fig,ax4)

        
        text = 'I made a Visualization Dashboard, using a kaggle dataset. 4 charts have been plotted from different types (Bar Chart,\
            Histogram, Line Chart, and Scatter Plot).'
        wrapped_text = textwrap.wrap(text, width=12)

        fig.text(0.92, 0.1, '\n'.join(wrapped_text), fontsize=12, bbox=dict(boxstyle='round', facecolor='none', edgecolor='grey', linewidth=2, pad=1, alpha=0.5), color='white')

        fig.suptitle("Student Adaptability Level in Online Education (FullName+StudentID)", fontsize=16, fontweight='bold', color='white')
        plt.savefig('StudentID.PNG', dpi=300, bbox_inches='tight')
        plt.close(fig)
        return fig
    

create_figure()