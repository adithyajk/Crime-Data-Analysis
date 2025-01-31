import pandas as pd
import numpy as np  
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as stats
import matplotlib.ticker as mtick
import sklearn.preprocessing as skp
import sklearn.model_selection as ms
import sklearn.calibration as sc
import math

input_file = r"C:\Users\adith\Downloads\Crime Data Analysis\Dataset\train.csv" #input file path
output_file=r"C:\Users\adith\Downloads\Crime Data Analysis\Output\updatedtrain34.csv" #output file path which is cleaned and updated.just cleaned not normalized
output_file1=r"C:\Users\adith\Downloads\Crime Data Analysis\Output\scaled_train15.csv" #output file path which is scaled and updated ie this normalized dataset 

# Function to drop columns if missing values exceed a threshold
def drop_column_if_high_missing(df, column_name, threshold=0.5):
    """Drops a column if the percentage of missing values is above the threshold."""
    if column_name in df.columns:
        missing_percentage = df[column_name].isnull().sum() / len(df)
        if missing_percentage > threshold:
            df = df.drop(columns=[column_name])
            print(f"Successfully dropped '{column_name}' column!\n")
    return df

# Function to fill missing values with a given value
def fill_missing_values(df, column_name, fill_value):
    """Fills missing values in a specified column with the equivalemt fill value ."""
    if column_name in df.columns:
        print(f"Filling missing values in '{column_name}' with '{fill_value}'...")
        df[column_name] = df[column_name].fillna(fill_value)
    return df

# Function to assign values to 'Weapon_desc', 'Weapon_cd', and 'Mode_cd' based on Crime_Category
def assign_crime_category_values(df):
    '''
    Categorizing the crimes based on the 'Crime_Category' column.
    Assign values to 'Weapon_desc', 'Weapon_cd', and 'Mode_cd' based on 'Crime_Category'by mapping the values to the respective categories.
    If the values are already present, they will be added as is.
    
    '''
    crime_category_mapping = {
        'Property Crimes': ('PC', 'PC_cd', 'PC_MO'),
        'Violent Crimes': ('VC', 'VC_cd', 'VC_MO'),
        'Crimes against Persons': ('CAP', 'CAP_cd', 'CAP_MO'),
        'Crimes against Public Order': ('CAPO', 'CAPO_cd', 'CAPO_MO'),
        'Fraud and White-Collar Crimes': ('FWCC', 'FWCC_cd', 'FWCC_MO'),
        'Other Crimes': ('OC', 'OC_cd', 'OC_MO')
    }

    for crime_category, (weapon_desc, weapon_cd, mode_cd) in crime_category_mapping.items():
        df.loc[(df['Crime_Category'] == crime_category) & (df['Weapon_Description'].isnull()), 'Weapon_desc'] = weapon_desc
        df.loc[df['Weapon_Description'].notnull(), 'Weapon_desc'] = df['Weapon_Description']
        df.loc[(df['Crime_Category'] == crime_category) & (df['Weapon_Used_Code'].isnull()), 'Weapon_cd'] = weapon_cd
        df.loc[df['Weapon_Used_Code'].notnull(), 'Weapon_cd'] = df['Weapon_Used_Code']
        df.loc[(df['Crime_Category'] == crime_category) & (df['Modus_Operandi'].isnull()), 'Mode_cd'] = mode_cd
        df.loc[df['Modus_Operandi'].notnull(), 'Mode_cd'] = df['Modus_Operandi']

    return df

# Function to handle 'Premise_Description' missing values based on 'Premise_Code'
def handle_premise_description(df):
    """Fill missing 'Premise_Description' values where 'Premise_Code' is 418."""
    df.loc[(df['Premise_Description'].isnull()) & (df['Premise_Code'] == 418), 'Premise_Description'] = 'UNKNOWN PREMISE'
    return df


#Function to detect and remove outliers
def detect_and_remove_outliers(df, column_name):
    """
    Detects and removes outliers in a specific column using the IQR method.

    Parameters:
        df (pd.DataFrame): The input DataFrame.
        column_name (str): The column to check for outliers.

    Returns:
        pd.DataFrame: The DataFrame with outliers removed.
    """
    Q1 = df[column_name].quantile(0.25)
    Q3 = df[column_name].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR

    # Print outlier info
    outliers = df[(df[column_name] < lower_bound) | (df[column_name] > upper_bound)]
    print(f"Outliers in '{column_name}':\n", outliers)

    # Remove outliers
    df_cleaned = df[(df[column_name] >= lower_bound) & (df[column_name] <= upper_bound)]
    return df_cleaned



def generate_crosstab(data, col1, col2):
    """
    Function to generate and display a cross-tabulation.
    
    Parameters:
    data (DataFrame): The dataset.
    col1 (str): The first column for cross-tabulation.
    col2 (str): The second column for cross-tabulation.
    
    Returns:
    pd.DataFrame: The cross-tabulation.
    """
    cross_tab = pd.crosstab(data[col1], data[col2])
    print(f"\nBivariate Frequency Table between {col1} and {col2}:\n")
    print(cross_tab.to_string())
    return cross_tab


def plot_heatmap(cross_tab, title, xlabel, ylabel):
    """
    Function to generate a heatmap from a cross-tabulation.

    Parameters:
    cross_tab (pd.DataFrame): The cross-tabulated data.
    title (str): The title of the heatmap.
    xlabel (str): Label for x-axis.
    ylabel (str): Label for y-axis.
    """
    plt.figure(figsize=(12, 8))
    ax = sns.heatmap(cross_tab, annot=True, fmt='d', cmap='coolwarm')
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.xticks(rotation=30, ha='right')  
    plt.yticks(rotation=0) 
    plt.tight_layout()
    plt.show()

    
  
def plot_stacked_bar(cross_tab, title, xlabel, ylabel):
    """
    Function to generate a grouped bar chart from a cross-tabulation and annotate with percentages.

    Parameters:
    cross_tab (pd.DataFrame): The cross-tabulated data.
    title (str): The title of the bar chart.
    xlabel (str): Label for x-axis.
    ylabel (str): Label for y-axis.
    """
    cross_tab_normalized = cross_tab.div(cross_tab.sum(axis=1), axis=0)  # Normalize to proportions
    ax = cross_tab_normalized.plot(kind='bar', figsize=(12, 8), colormap='viridis', width=0.8)

    # Annotate each bar with percentages
    for container in ax.containers:  # Iterate over each container (group of bars)
        for bar in container:  # Iterate over each bar within the container
            height = bar.get_height()
            if height > 0:  # Annotate only if the bar has height
                ax.annotate(f'{height*100:.2f}%',
                            (bar.get_x() + bar.get_width() / 2, height),
                            ha='center', va='bottom', fontsize=9, color='black')

    # Add labels and formatting
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel + " (%)")
    plt.legend(title='Categories', bbox_to_anchor=(1.05, 1), loc='upper left')
    ax.yaxis.set_major_formatter(mtick.PercentFormatter(xmax=1))  # Ensure percentages are on the y-axis
    plt.xticks(rotation=30, ha='right')  # Adjust x-axis label rotation
    plt.tight_layout()
    plt.show()


def chi_square_test(cross_tab):
    """
    Function to perform a chi-square test for independence.
    
    Parameters:
    cross_tab (pd.DataFrame): The cross-tabulated data.
    
    Returns:
    tuple: Chi-square statistic, p-value, degrees of freedom, expected frequencies
    """
    chi2, p, dof, ex = stats.chi2_contingency(cross_tab)
    print(f"\nChi-Square Test Results:")
    print(f"Chi2 Statistic: {chi2}")
    print(f"P-Value: {p}")
    print(f"Degrees of Freedom: {dof}")
    
    if p < 0.05:
        print("Conclusion: There is a significant association between the variables.")
    else:
        print("Conclusion: There is no significant association between the variables.")
    
    return chi2, p, dof, ex

def plot_correlation_heatmap(data, num_cols, title):
    """
    Function to generate a correlation heatmap for numerical variables.
    
    Parameters:
    data (pd.DataFrame): The dataset containing the numerical columns.
    num_cols (list): List of numerical columns to include in the correlation heatmap.
    title (str): The title of the heatmap.
    """
    corr_matrix = data[num_cols].corr()
    plt.figure(figsize=(12, 7))
    ax = sns.heatmap(corr_matrix, annot=True, vmin=-1, vmax=1, cmap='coolwarm', cbar_kws={'label': 'Correlation Coefficient'})
    plt.title(title)
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.show()

#function univariant...................

def analyze_numerical_columns(data, num_cols_of_interest):
    """
    Perform univariate analysis on specified numerical columns, displaying results as percentages.
   
    Parameters:
        data (pd.DataFrame): The dataset.
        num_cols_of_interest (list): List of column names for analysis.
    """
    # Ensure the columns exist in the dataset
    existing_cols = [col for col in num_cols_of_interest if col in data.columns]
   
    for col in existing_cols:
        print(f'Analyzing column: {col}')
        print('Skewness:', round(data[col].skew(), 2))
       
        # Normalize data to percentages for the histogram (scale from 0 to 100)
        max_value = data[col].max()
        data_percentage = (data[col] / max_value) * 100  # Convert to percentage
       
        # Create subplots for histogram and boxplot
        plt.figure(figsize=(14, 11))
       
        # Plot the normalized histogram
        plt.subplot(1, 2, 1)
        n, bins, patches = plt.hist(data_percentage, bins=20, color='skyblue', edgecolor='black', density=True)
       
        # Annotate the bars with percentages (rotated by 90 degrees to the left)
        offset=0.0005
        for i in range(len(patches)):
            height = patches[i].get_height()
            x_position = patches[i].get_x() + patches[i].get_width() / 2
            # Format percentage and annotate
            plt.annotate(f'{height*100:.2f}%', (x_position, height + offset), ha='center', va='bottom',
                         fontsize=8, color='black', fontweight='bold', rotation=90)
       
        # Title and axis labels
        plt.title(f'Normalized Histogram of {col} (Percentage)',y=1.005)
        # #increase plt.title height
        # plt.title(f'Normalized Histogram of {col} (Percentage)', y=1.05)
        plt.xlabel(f'{col} (Percentage)', fontsize=12)
        plt.ylabel('Percentage', fontsize=12)
        
        import matplotlib.ticker as mticker
        # Set x-ticks to show percentage labels (0 to 100)
        plt.xticks(np.arange(0, 110, 10))  # From 0% to 100% in steps of 10
        plt.gca().yaxis.set_major_locator(mticker.MaxNLocator(nbins=10)) 
       
        # Plot the boxplot (normalize the column to percentage scale)
        plt.subplot(1, 2, 2)
        sns.boxplot(x=data_percentage, color='lightgreen')
       
        # Get statistical data
        median = np.median(data_percentage)
        lower_q, upper_q = np.percentile(data_percentage, [25, 75])
        iqr = upper_q - lower_q
        lower_bound = lower_q - 1.5 * iqr
        upper_bound = upper_q + 1.5 * iqr
        outliers = data_percentage[(data_percentage < lower_bound) | (data_percentage > upper_bound)]

        # Annotate the boxplot for clarification
        plt.title(f'Boxplot of {col} (Percentage)')
       
        # Annotate Median
        plt.annotate(f'Median: {median:.2f}%', xy=(0, median), xytext=(10, median + 5),
                     arrowprops=dict(facecolor='black', shrink=0.05), fontsize=12, color='black')

        # Annotate Quartiles
        plt.annotate(f'Q1 (25th percentile): {lower_q:.2f}%', xy=(0, lower_q), xytext=(10, lower_q - 5),
                     arrowprops=dict(facecolor='black', shrink=0.05), fontsize=12, color='black')
        plt.annotate(f'Q3 (75th percentile): {upper_q:.2f}%', xy=(0, upper_q), xytext=(10, upper_q + 5),
                     arrowprops=dict(facecolor='black', shrink=0.05), fontsize=12, color='black')

        # Annotate Outliers
        for outlier in outliers:
            plt.annotate(f'Outlier: {outlier:.2f}%', xy=(0, outlier), xytext=(10, outlier),
                         arrowprops=dict(facecolor='red', shrink=0.05), fontsize=12, color='red')
       
        # Show the plots
        plt.tight_layout()
        plt.show()


def plot_categorical_bar(data, cat_col, fig_title="Bar plot for categorical variable"):
    """
    Create a bar plot for a single categorical variable, displaying percentages.


    Parameters:
        data (pd.DataFrame): The dataset.
        cat_col (str): The categorical column name.
        fig_title (str): Title for the figure.
    """
    # Create the bar plot for a single categorical variable
    plt.figure(figsize=(10, 6))
    ax = sns.countplot(x=cat_col, data=data, color='blue', stat='percent',
                       order=data[cat_col].value_counts().index)
   
    # Set the title and labels
    ax.set_title(fig_title, fontsize=16)
    ax.set_ylabel('Percentage')
   
    # Annotate the bars with percentages
    total = len(data[cat_col])
    offset=1.0
    for p in ax.patches:
        height = p.get_height()
        ax.annotate(f'{height:.2f}%',  # Format to 2 decimal places
                    (p.get_x() + p.get_width() / 2., height + offset),
                    ha='center', va='center', fontsize=8,fontweight='bold', color='black')


    # Set y-axis limit based on the maximum percentage value (so it doesn't always go up to 100)
    max_percentage = data[cat_col].value_counts(normalize=True).max() * 100
    ax.set_ylim(0, max_percentage + 10)  # Add a small margin above the max percentage


    # Rotate x-axis labels for better readability
    ax.tick_params(labelrotation=45)
   
    # Show the plot
    plt.tight_layout()
    plt.show()

#PAIR PLOT...........................

def plot_pairplot(dataset_path, hue_column='Crime_Category'):
    """
    Function to create a Seaborn pair plot for relevant numerical columns in a dataset.

    Parameters:
    - dataset_path: str, path to the CSV file containing the data.
    - hue_column: str, the column name to be used for color-coding (default is 'Crime_Category').
    """
    # Load dataset
    train = pd.read_csv(dataset_path)

    # Display columns and data
    print("Columns in dataset:", train.columns)
    print("First few rows:\n", train.head())

    # Define columns suitable for pair plotting
    # Only include numerical columns and exclude irrelevant or ID-like columns
    numerical_columns = train.select_dtypes(include=["number"]).columns
    columns_to_pairplot = []

    # Iterate through numerical columns and include them based on justification
    for col in numerical_columns:
        if col == 'Victim_Age':
            print(f"Including column '{col}' for pair plotting: Reason - Represents demographic data that could correlate with other features.")
            columns_to_pairplot.append(col)
        elif col == 'Time_Occurred':
            print(f"Including column '{col}' for pair plotting: Reason - Represents temporal data that might relate to other features like location or area.")
            columns_to_pairplot.append(col)
        elif col == 'Area_ID':
            print(f"Including column '{col}' for pair plotting: Reason - Represents LAPD Geographic Area, useful for analyzing crime distribution.")
            columns_to_pairplot.append(col)
        elif col == 'Reporting_District_no':
            print(f"Excluding column '{col}': Reason - High-cardinality ID-like feature with little analytical value for relationships.")
        elif col == 'Premise_Code':
            print(f"Including column '{col}' for pair plotting: Reason - Represents categories relevant to crime context.")
            columns_to_pairplot.append(col)

    # Add the hue column if it exists
    if hue_column in train.columns:
        print(f"Using '{hue_column}' as the hue for pair plotting.")
        if hue_column not in columns_to_pairplot:
            columns_to_pairplot.append(hue_column)

    # Ensure at least two columns are selected for the pair plot
    if len(columns_to_pairplot) > 1:
        sns.pairplot(train[columns_to_pairplot], hue=hue_column)
    else:
        print("Not enough relevant columns for pair plotting.")

    plt.show()
    


def visualize_crime_category_distribution_percentage(y_train, y_test):
    """ 
    Visualize the percentage distribution of crime categories in train and test datasets.

    Parameters:
        y_train (pd.Series): The target variable from the training data.
        y_test (pd.Series): The target variable from the testing data.
    """
    # Calculate percentages for each category in train and test
    train_percentage = y_train.value_counts(normalize=True) * 100
    test_percentage = y_test.value_counts(normalize=True) * 100

    # Combine percentages into a single DataFrame
    percentage_df = pd.DataFrame({
        'Train (%)': train_percentage,
        'Test (%)': test_percentage
    }).sort_index()

    # Plot the grouped bar chart
    ax = percentage_df.plot(kind='bar', figsize=(12, 6), color=['skyblue', 'orange'], width=0.8)

    # Customize plot appearance
    plt.title('Percentage Distribution of Crime Categories (Train vs Test)', fontsize=16)
    plt.xlabel('Crime Category', fontsize=12)
    plt.ylabel('Percentage (%)', fontsize=12)
    plt.xticks(rotation=45, ha='right', fontsize=10)
    plt.yticks(fontsize=10)
    plt.legend(title="Dataset", fontsize=10)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()

    # Annotate percentage on top of each bar
    for p in ax.patches:
        height = p.get_height()
        ax.annotate(f'{height:.2f}%', 
                    (p.get_x() + p.get_width() / 2., height), 
                    ha='center', va='center', fontsize=10, color='black', xytext=(0, 5), textcoords='offset points')

    # Show the plot
    plt.show()

def plot_grouped_data(df, group_columns, plot_title, x_col, y_col, hue_col, plot_type="bar"):
    # Step 1: Group by the provided columns
    grouped_data = df.groupby(group_columns).size().reset_index(name=y_col)
    
    # Step 2: Plot the grouped data
    plt.figure(figsize=(14, 10))
    ax = None
    if plot_type == "bar":
        ax = sns.barplot(x=x_col, y=y_col, hue=hue_col, data=grouped_data, palette='pastel', ci=None)
    else:
        ax = sns.lineplot(x=x_col, y=y_col, hue=hue_col, data=grouped_data, marker='o')
    
    # Set plot title and labels
    plt.title(plot_title)
    plt.xlabel(x_col)
    plt.ylabel(y_col)
    plt.xticks(rotation=45, ha='right')
    plt.legend(title=hue_col, bbox_to_anchor=(1.05, 1), loc='upper left')

    total = grouped_data[y_col].sum()
    if total == 0:
        total = 1  # Prevent division by zero

    for p in ax.patches:
        height = p.get_height()
        percentage = (height / total) * 100

        if percentage >0.00:  # Show very small percentages as well
            xytext_offset = 10 if height > 5 else 3  # Adjust vertical position dynamically
            ax.annotate(f'{percentage:.2f}%', 
                        (p.get_x() + p.get_width() / 2., height), 
                        ha='center', va='bottom', fontsize=8, color='black', 
                        xytext=(0, xytext_offset), textcoords='offset points')

        else:
            for line in ax.lines:
                x_data, y_data = line.get_data()
                for x, y in zip(x_data, y_data):
                    ax.annotate(f'{y:.2f}', (x, y), ha='center', va='bottom', fontsize=10, color='black')

    # Show the plot
    plt.tight_layout()
    plt.show()
    
    
def plot_grouped_data_with_rotated_annotations(df, group_columns, plot_title, x_col, y_col, hue_col, plot_type="bar"):
            # Step 1: Group by the provided columns
            grouped_data = df.groupby(group_columns).size().reset_index(name=y_col)

            # Step 2: Plot the grouped data
            plt.figure(figsize=(14, 10))
            ax = None
            if plot_type == "bar":
                ax = sns.barplot(
                    x=x_col, y=y_col, hue=hue_col, data=grouped_data, 
                    palette='pastel', ci=None, dodge=True
                )
            else:
                ax = sns.lineplot(x=x_col, y=y_col, hue=hue_col, data=grouped_data, marker='o')

            # Set plot title and labels
            plt.title(plot_title)
            plt.xlabel(x_col)
            plt.ylabel(y_col)
            plt.xticks(rotation=45, ha='right')
            plt.legend(title=hue_col, bbox_to_anchor=(1.05, 1), loc='upper left')

            # Annotate each bar with its percentage for bar plots
            if plot_type == "bar":
                total = grouped_data[y_col].sum()  # Get the total sum for percentage calculation
                for p in ax.patches:
                    height = p.get_height()
                    if height > 0:  # Avoid division by zero
                        percentage = (height / total) * 100
                        ax.annotate(
                            f'{percentage:.2f}%', 
                            (p.get_x() + p.get_width() / 2., height), 
                            ha='center', va='bottom', 
                            fontsize=9, color='black', 
                            rotation=90,  # Rotate text 90 degrees
                            xytext=(0, 5), textcoords='offset points'
                        )

            # Show the plot
            plt.tight_layout()
            plt.show()
            
# Main function to process the dataset

try:
        data = pd.read_csv(input_file)
        if data.empty:
            print("The file is empty. Please check the dataset.")
        else:
            print("Data read successfully!\n")
            
            
            ##converting Time_Occurred to dateime datatype
            # Add the Time_Occurred transformation here
            data['Time_Occurred'] = data['Time_Occurred'].astype(str).str.zfill(4)
            data['Time_Occurred'] = pd.to_datetime(data['Time_Occurred'], format='%H%M').dt.time
            
           #Slicing the the String from 'Date_Reported' and 'Date_Occured' column to get only the date
            data['Date_Reported'] = data['Date_Reported'].str.slice(0, 10)
            data['Date_Occurred'] = data['Date_Occurred'].str.slice(0, 10)    
        

            # Drop 'Cross_Street' column if it has more than 50% missing values
            data = drop_column_if_high_missing(data, 'Cross_Street', threshold=0.5)

            # Fill missing values in 'Victim_Sex' and 'Victim_Descent' with 'X'
            data = fill_missing_values(data, 'Victim_Sex', 'X')
            data = fill_missing_values(data, 'Victim_Descent', 'X')

         
        
            
            # Assign values based on Crime_Category
            data = assign_crime_category_values(data)
            
            #Drop 'Weapon_Description', 'Weapon_Used_Code', and 'Modus_Operandi' columns
            data = drop_column_if_high_missing(data, 'Weapon_Description', threshold=0.5)
            data = drop_column_if_high_missing(data, 'Weapon_Used_Code', threshold=0.5)
            data = drop_column_if_high_missing(data, 'Modus_Operandi', threshold=0.13)

            # Handle missing 'Premise_Description' values based on 'Premise_Code'
            data = handle_premise_description(data)
            
            
            # Identify numerical columns
            numerical_columns = data.select_dtypes(include=['int64', 'float64']).columns

            # Detect and remove outliers for each numerical column
            #drop the entire row if victime age is less than 0s
            data = data[data['Victim_Age'] > 0]
            for col in numerical_columns:
                print(f"Checking for outliers in column: {col}")
                data = detect_and_remove_outliers(data, col)
            

            # Display updated number of missing values
            print("\nUpdated number of missing values in each column:\n")
            print(data.isnull().sum())
            print("\nUpdated Data shape:", data.shape)
            # # print("\nDisplaying the complete updated dataset:\n",data.to_string())
            try: #Saving the updated dataset
                data.to_csv(output_file, index=False)
                print(f"\nThe updated dataset has been saved to: {output_file}\n")
            except Exception as e:
                print(f"An error occurred while saving the updated dataset: {e}")


             # Numerical columns for univariate analysis
            num_cols_of_interest = ['Premise_Code', 'Victim_Age','Area_ID']
            analyze_numerical_columns(data, num_cols_of_interest)

            # Categorical columns for bar plots
            # cat_cols = ['Victim_Sex', 'Victim_Descent', 'Crime_Category']
            plot_categorical_bar(data,'Victim_Sex')
            plot_categorical_bar(data, 'Crime_Category')
            plot_categorical_bar(data, 'Victim_Descent')
            plot_categorical_bar(data,'Area_Name')
            plot_categorical_bar(data,'Status_Description')

            #Bivariate Analysis
            #Area_Name vs Crime_Category
            cross_tab_area_crime = generate_crosstab(data, 'Area_Name', 'Crime_Category')
            plot_heatmap(cross_tab_area_crime, 'Bivariate Analysis between Area_Name and Crime_Category', 'Crime_Category', 'Area_Name')
            plot_stacked_bar(cross_tab_area_crime, 'Stacked Bar Chart between Area_Name and Crime_Category', 'Area', 'Proportion')
            chi_square_test(cross_tab_area_crime)
            #Weapon_cd vs Crime_Category
            cross_tab_weapon_crime = generate_crosstab(data, 'Weapon_cd', 'Crime_Category')
            chi_square_test(cross_tab_weapon_crime)
            #Mode_cd vs Crime_Category
            cross_tab_victim_sex = generate_crosstab(data, 'Victim_Sex', 'Crime_Category')
            plot_heatmap(cross_tab_victim_sex, 'Bivariate Analysis between Victim_Sex and Crime_Category', 'Crime_Category', 'Victim_Sex')
            plot_stacked_bar(cross_tab_victim_sex, 'Stacked Bar Chart between Victim_Sex and Crime_Category', 'Victim_Sex', 'Proportion')
            chi_square_test(cross_tab_victim_sex)
            #Victim_Age vs Crime_Category
            cross_tab_age_crime = generate_crosstab(data, 'Victim_Age', 'Crime_Category')
            chi_square_test(cross_tab_age_crime)
            #Status_Description vs Crime_Category
            cross_tab_status_crime = generate_crosstab(data, 'Status_Description', 'Crime_Category')
            plot_heatmap(cross_tab_status_crime, 'Bivariate Analysis between Status_Description and Crime_Category', 'Crime_Category', 'Status_Description')
            plot_stacked_bar(cross_tab_status_crime, 'Stacked Bar Chart between Status_Description and Crime_Category', 'Status_Description', 'Proportion')
            chi_square_test(cross_tab_status_crime)

            
           # Multivariate analysis: Correlation heatmap
            num_cols1 = ['Area_ID','Reporting_District_no',	'Part 1-2','Premise_Code','Victim_Age']  
            plot_correlation_heatmap(data, num_cols1, 'Correlation Heatmap of Numerical Variables')




            #pair plotting
            plot_pairplot(output_file)
            
            #GROUP BY
            # Load the dataset
            df = pd.read_csv(output_file)   


           # Create Victim Age Groups
            age_bins = [0, 10, 20, 30, 40, 50, 100]
            age_labels = ['0-10', '10-20', '20-30', '30-40', '40-50', '50+']
            df['Victim_Age_Group'] = pd.cut(df['Victim_Age'], bins=age_bins, labels=age_labels, right=False)

            # Create Area_ID Groups
            area_bins = [1, 5, 10, 15, 20, 25]
            area_labels = ['1-5', '5-10', '10-15', '15-20', '20-25']
            df['Area_ID_Group'] = pd.cut(df['Area_ID'], bins=area_bins, labels=area_labels, right=False)

            # Create Time Occurred Groups
            df['Time_Occurred'] = pd.to_timedelta(df['Time_Occurred'].astype(str))
            time_bins = [0, 18000, 36000, 54000, 72000, 86400]
            time_labels = ['0:01:00-5:00:00', '5:00:00-10:00:00', '10:00:00-15:00:00', '15:00:00-24:00:00', '24:00:00-30:00:00']
            df['Time_Occurred_Group'] = pd.cut(df['Time_Occurred'].dt.total_seconds(), bins=time_bins, labels=time_labels, right=False)

            # Group 1: Crime_Category, Victim_Age_Group, Area_ID_Group, Time_Occurred_Group
            plot_grouped_data(df, 
                            ['Crime_Category', 'Victim_Age_Group', 'Area_ID_Group', 'Time_Occurred_Group'], 
                            'Crime Category Count by Victim Age Group, Area ID Group, and Time Occurred Group',
                            'Crime_Category', 'Count', 'Victim_Age_Group')

            # Group 2: Victim_Sex, Victim_Age_Group, Time_Occurred_Group
            plot_grouped_data(df, 
                            ['Victim_Sex', 'Victim_Age_Group', 'Time_Occurred_Group'], 
                            'Crime Count by Victim Sex, Victim Age Group, and Time Occurred Group',
                            'Victim_Sex', 'Count', 'Victim_Age_Group')

            # Group 3: Victim_Age_Group, Area_ID_Group, Time_Occurred_Group
            plot_grouped_data(df, 
                            ['Victim_Age_Group', 'Area_ID_Group', 'Time_Occurred_Group'], 
                            'Crime Count by Victim Age Group, Area ID Group, and Time Occurred Group',
                            'Victim_Age_Group', 'Count', 'Area_ID_Group')

            # Group 4: Area_Name, Victim_Age_Group
            plot_grouped_data(df, 
                            ['Area_Name', 'Victim_Age_Group'], 
                            'Crime Count by Area Name and Victim Age Group',
                            'Area_Name', 'Count', 'Victim_Age_Group')

            # Group 5: Victim_Descent, Victim_Age_Group
           
            plot_grouped_data_with_rotated_annotations(df, 
            ['Victim_Descent', 'Victim_Age_Group'], 
            'Crime Count by Victim Descent and Victim Age Group',
            'Victim_Descent', 'Count', 'Victim_Age_Group')
      

            # Generate descriptive statistics for numeric columns
            descriptive_stats = data.describe()

            # Show the descriptive statistics
            print(descriptive_stats)
              
            #Standardization of numerical columns
          
            numerical_columns = data.select_dtypes(include=['number']).columns
            standardized_data = data.copy()  # Create a copy to preserve the original data
            standardized_data[numerical_columns] = (data[numerical_columns] - data[numerical_columns].mean()) / data[numerical_columns].std()
            data1=data
            data=standardized_data
            
            #performing label encoding
            label_encoder = sc.LabelEncoder()
            columns_to_encode=['Victim_Sex', 'Area_Name', 'Victim_Descent', 'Crime_Category', 'Premise_Description', 'Weapon_desc', 'Status', 'Status_Description']
            for column in columns_to_encode:
                data[column] = label_encoder.fit_transform(data[column])

            #saving dataset to csv
            try:
                data.to_csv(output_file1, index=False)
                print(f"\nThe updated dataset has been saved to: {output_file1}\n")
            except Exception as e:
                print(f"An error occurred while saving the updated dataset: {e}")
            
            print(data.head())
            
            
        # Define features (X) and target variable (y)
        X = data.drop(columns=['Crime_Category'])  # Features
        y = data['Crime_Category']                # Target

        # Split data: 80% train, 20% test
        X_train, X_test, y_train, y_test = ms.train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
      
        # Columns to scale
        min_max_cols = ['Latitude', 'Longitude', 'Reporting_District_no']  
        standardize_cols = ['Victim_Age', 'Area_ID', 'Premise_Code']    
        robust_cols = ['Premise_Code', 'Part 1-2']

        # Initialize scalers
        min_max_scaler = skp.MinMaxScaler()
        standard_scaler = skp.StandardScaler()
        robust_scaler = skp.RobustScaler()

        # Apply scalers only to the training set, and transform both training and test sets
        X_train[min_max_cols] = min_max_scaler.fit_transform(X_train[min_max_cols])
        X_test[min_max_cols] = min_max_scaler.transform(X_test[min_max_cols])

        X_train[standardize_cols] = standard_scaler.fit_transform(X_train[standardize_cols])
        X_test[standardize_cols] = standard_scaler.transform(X_test[standardize_cols])

        X_train[robust_cols] = robust_scaler.fit_transform(X_train[robust_cols])
        X_test[robust_cols] = robust_scaler.transform(X_test[robust_cols])
        
        

 
        #print the data
        print("\nScaled Training Data:")
        print(X_train.head())
        print("\nScaled Testing Data:")
        print(X_test.head())
            

    

    #Calling for seeing differnce in distribution of crime category in train and test data
        visualize_crime_category_distribution_percentage(y_train, y_test)

       






except FileNotFoundError:
        print(f"Error: File not found at the specified path: {input_file}")
except Exception as e:
        print(f"An unexpected error occurred: {e}")

