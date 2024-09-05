import pandas as pd
import tabula

# Read the table from page 65
df_copper_production = tabula.read_pdf("C:/Users/sadoo/OneDrive/Desktop/commodity_prices_new/data/Factbook2023.pdf", pages=65, multiple_tables=True)[0]

# Rename columns
df_copper_production = df_copper_production.rename(columns={
    df_copper_production.columns[0]: 'Year1',
    df_copper_production.columns[2]: 'Production1',
    df_copper_production.columns[3]: 'Usage1',
    df_copper_production.columns[4]: 'Year2',
    df_copper_production.columns[6]: 'Production2',
    df_copper_production.columns[7]: 'Usage2',
    df_copper_production.columns[8]: 'Year3',
    df_copper_production.columns[10]: 'Production3',
    df_copper_production.columns[11]: 'Usage3',
})

# Drop specified columns
df_copper_production = df_copper_production.drop(columns=[
    df_copper_production.columns[1],
    df_copper_production.columns[5],
    df_copper_production.columns[9]
]).reset_index(drop=True)

# Drop rows with NaN values
df_copper_production = df_copper_production.dropna()

# Change specific rows in column Year3
df_copper_production.loc[1, 'Year3'] = 2002
df_copper_production.loc[22, 'Year3'] = 2022

# Convert Year columns to integers
year_columns = ['Year1', 'Year2', 'Year3']
for col in year_columns:
    df_copper_production[col] = df_copper_production[col].astype(float).astype(int)

# Stack the columns
df_stacked = pd.DataFrame({
    'Year': df_copper_production['Year1'].tolist() + 
            df_copper_production['Year2'].tolist() + 
            df_copper_production['Year3'].tolist(),
    'Production': df_copper_production['Production1'].tolist() + 
                  df_copper_production['Production2'].tolist() + 
                  df_copper_production['Production3'].tolist(),
    'Usage': df_copper_production['Usage1'].tolist() + 
             df_copper_production['Usage2'].tolist() + 
             df_copper_production['Usage3'].tolist()
})

# Sort the stacked dataframe by Year
df_stacked = df_stacked.sort_values('Year').reset_index(drop=True)

# Convert Year to datetime
df_stacked['Year'] = pd.to_datetime(df_stacked['Year'], format='%Y').dt.year

# Convert Production and Usage columns to integers
df_stacked['Production'] = df_stacked['Production'].str.replace(',', '').astype(int)
df_stacked['Usage'] = df_stacked['Usage'].str.replace(',', '').astype(int)

# Set Year as index
df_stacked.set_index('Year', inplace=True)

# Output to CSV file for Streamlit app
df_stacked.to_csv('copper_inventory.csv')

print("CSV file saved as 'copper_inventory.csv'.")
