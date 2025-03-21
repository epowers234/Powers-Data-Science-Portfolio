# TidyData-Project

## Project Overview

Cleaning data is notoriously one of the most challenging parts of data analysis. It is important to get data into a standard format that the computer can easily understand when creating visualizations and analyses. One part of this is Tidy Data, which puts each variable in its column and each observation in its row. To learn how to apply this, I am taking an "untidy" data set that compiles the medalists from the 2008 Olympics. I aim to tidy this data set and then generate multiple visualizations and pivot tables to conclude the differences present in the data set. This will teach me how to apply tidy data principles and how it aids in the analysis process. 

## Table of Contents

1. [Project Overview](#project-overview)
2. [Installation and Setup](#installation-and-setup)
3. [Codes and Resources Used](#codes-and-resources-used)
4. [Data](#data)
5. [Data Processing and Analysis](#data-processing-and-analysis)
6. [Results and Visualizations](#results-and-visualizations)
7. [Future Work](#future-work)

## Installation and Setup

Jupyter Notebooks was used as the platform for Python code. I used the Anaconda Navigator to access Jupyter Notebooks, but other programs, such as Google Colab, can also be used to access the files.

## Codes and Resources Used

1.   **Tidy Data Cheat Sheet**
   - Reference sheet: [Pandas Cheat Sheet](https://pandas.pydata.org/Pandas_Cheat_Sheet.pdf).
   - Aided when performing the Tidy Data commands at the start.
2.   **Purpose of Tidy Data**
   - Visit [Tidy Data by Hadley Wickham](https://vita.had.co.nz/papers/tidy-data.pdf).
   - Provides background on the importance of Tidy Data.
   - Cited in my description of the purpose in my Jupyter File.  
3.   **From Data to Viz as a starting point for data visualization**
   - Visit [From Data to Viz](https://www.data-to-viz.com/).
   - Code derived from the stacked bar graph and circular bar graph sections.
4.   **Necessary libraries that need to be imported**
   - Pandas
   - Numpy
   - Matplotlib.pyplot
   - Matplotlib

## Data

The dataset used in this project is initially untidy, but it contains each medalist's name, what type of medal received, and in what event that is sorted by gender. 

The dataset used in this project was sourced from [2008 Medalists Original Source](https://edjnet.github.io/OlympicsGoNUTS/2008/). It was adapted to this untidy source [olympics_08_medalists.csv](https://github.com/user-attachments/files/19287715/olympics_08_medalists.csv). Ensure you download the dataset and place it in the appropriate data directory within the project folder.

## Data Processing and Analysis

1.   **Tidying the data**
   - Used the pandas library. 
   - Followed basic tidy data commands such as df.melt, df.dropna(), df.str.split, df.str.replace, and df.rename.
2.   **Generating a stacked barplot**
   - Used From Data to Viz as a starting point.
   - Used numpy, matplotlib.pyplot, matplotlib, and pandas libraries.
   - Created a barplot that separates female and male athletes and depicts the number of medals awarded across the three types: gold, silver, and bronze.  
3.   **Creating circular barplots**
   - Used From Data to Viz as a starting point.
   - Used pandas, matplotlib.pyplot, and numpy libraries.
   - Created two circular barplots, one for male and one for female participants, and presented each event and the number of awarded medals.
4.   **Creating pivot tables**
   - Used pandas library.
   - Made two pivot tables.
   - One counts the total number of medals for both genders in each event.
   - The second shows the distribution of those medals in each event by bronze, gold, and silver.

## Results and Visualizations
### **Stacked barplot**
<img width="341" alt="Screenshot 2025-03-21 at 1 16 17 PM" src="https://github.com/user-attachments/assets/80d54099-2e65-43f8-a3c8-cbd6d38dd23b" />
<br /> Showed that more medals are awarded to male participants than female participants overall. 

### **Circular barplots**
#### Male: 
<img width="502" alt="Screenshot 2025-03-21 at 1 16 28 PM" src="https://github.com/user-attachments/assets/01d6813b-bf9b-4924-8da7-01ece3fe0ca5" />

#### Female: 
<img width="509" alt="Screenshot 2025-03-21 at 1 16 34 PM" src="https://github.com/user-attachments/assets/b38ca0be-1218-49d1-aac5-e3e018501911" />

<br /> These two graphs provide insight individually by demonstrating what events are offered and how many athletes participate by gender. Then, one can compare both of them to see the differences on what is offered by gender in the 2008 Olympics. 

### **Pivot tables**
#### Total medal count for both genders:
<img width="169" alt="Screenshot 2025-03-21 at 1 19 39 PM" src="https://github.com/user-attachments/assets/23605407-579f-41d9-9d5b-ddb775d9d86a" />

#### Total medal count for both genders broken down by medal type: 
<img width="265" alt="Screenshot 2025-03-21 at 1 19 45 PM" src="https://github.com/user-attachments/assets/08c622df-9e2e-4471-a02d-6db03f747d34" />

<br /> These tables are a way to visualize and understand how different events have a different number of awarded medals for the overall games, not differentiated by gender. 

## Future Work

- Explore why certain events have differing amounts of medals awarded.
- Analyze past and future Olympic games to see how these medal counts have changed. 
- Apply Tidy Data principles to more expansive and complex data sets. 
