#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 20 11:21:53 2023

@author: Yannick
"""

#Load packages 
import numpy as np
import pandas as pd
import geopandas as gpd
import plotly.express as px
from dash import Dash, dcc, html
from dash.dependencies import Input, Output
import dash_bootstrap_components as dbc

#Load data
soy_publications = pd.read_csv("soy_europe.csv") #Soy publications on Europe
import_data = pd.read_excel("ds-018995_page_spreadsheet-4.xlsx", #import data of soy
                                   sheet_name = "Sheet 1", 
                                   header=None, 
                                   skiprows=8,
                                   skipfooter=3) 
gdf_world = gpd.read_file("ne_10m_admin_0_countries/ne_10m_admin_0_countries.shp") #Polygons of the world
export_data = pd.read_excel("ds-018995_page_spreadsheet-5.xlsx", # export data of soy
                                   sheet_name = "Sheet 1", 
                                   header=None, 
                                   skiprows=8,
                                   skipfooter=3) 

production_data = pd.read_excel("apro_cpsh1__custom_8791857_spreadsheet.xlsx", #own production of soy
                                   sheet_name = "Sheet 1", 
                                   header=None, 
                                   skiprows=8,
                                   skipfooter=8) 

#%%
EU_27 = ['AT', 'BE', 'BG', 'CY', 'CZ', 'DE', 'DK', 'EE', 'ES', 'FI', 'FR', 'GR', 'HR', 'HU', 'IE', 'IT', 'LT', 'LU', 'LV', 'MT', 'NL', 'PL', 'PT', 'RO', 'SE', 'SI', 'SK'] #Country codes EU27

#Add an ISO_alpha_2 value to namibia
import_data.loc[0, import_data.loc[1, :] == "Namibia"] = "NA"
export_data.loc[0, export_data.loc[1, :] == "Namibia"] = "NA"

#change the same for Greece to GR
production_data.loc[production_data[1] == "Greece", 0] = "GR"

#Find all countries
temp_vec = import_data.iloc[0, :].unique() #Country codes
countries = pd.DataFrame(temp_vec[1:], columns=['Country_code'])
temp_vec = import_data.iloc[1, :].unique() #country names
country_names_df = pd.DataFrame(temp_vec[1:], columns=['Country_name'])
names_codes = pd.concat([countries, country_names_df], axis=1)

#Filter out EU countries
countries_extraEU = countries[~countries['Country_code'].isin(EU_27)]

#Make the index be the years
production_data.columns = production_data.loc[0, :]

#Find the years
years = itemp_vec = import_data.iloc[4:, 1]

#Filter only EU countries
production_EU = production_data[production_data.iloc[:,0].isin(EU_27)].reset_index(drop=True)
production_EU.index = production_EU.iloc[:, 0]

#Select all but 2 columns
production_EU_clean = production_EU.iloc[:, 2:]
production_EU_clean.columns = production_EU_clean.columns.astype(int)
production_EU_clean = production_EU_clean.apply(pd.to_numeric, errors='coerce')

#%%
# Create an empty dictionary to store DataFrames for each country
import_dfs = {}
export_dfs = {}

#initiate empty dfs for import
import_tot = pd.DataFrame()
import_beans = pd.DataFrame()
import_oil = pd.DataFrame()
import_meal = pd.DataFrame()

#initiate empty dfs for export
export_tot = pd.DataFrame()
export_beans = pd.DataFrame()
export_oil = pd.DataFrame()
export_meal = pd.DataFrame()

#For loop to create dataframe for the imports over time
for value in countries_extraEU['Country_code']:
    mask = (import_data.iloc[0, :] == value) #Create a mask to filter the dataframes
    filtered_data = import_data.loc[:, mask] #filter the data for the specific country
    filtered_data.columns = filtered_data.iloc[2,:]  #Change the colnames to represent the soy flow
    clean_data = filtered_data.iloc[4:,:] #Select the useful rows
    clean_data = clean_data.replace(":", np.nan) #Make the missing values be represented as nan
    clean_data = clean_data.apply(pd.to_numeric, errors='coerce') #Make all cells be represented as numeric
    clean_data = clean_data.set_index(years)
    import_dfs[value] = clean_data #Save the df in a dictionary
    total_import = clean_data.sum(axis=1) #sum the import
    oil_import = clean_data['Oilcake & other solid residues of oil from soya beans'] #Select the different imports
    bean_import = clean_data['Soya beans'] #Select the different imports
    meal_import =clean_data['Soya bean oil and its fractions'] #Select the different imports
    import_tot[value] = total_import #Append the df
    import_beans[value] = bean_import #Append the df
    import_oil[value] = oil_import #Append the df
    import_meal[value] = meal_import #Append the df

#For loop to create dataframe for the exports over time
for value in countries_extraEU['Country_code']:
    mask = (export_data.iloc[0, :] == value) #Create a mask to filter the dataframes
    filtered_data = export_data.loc[:, mask] #filter the data for the specific country
    filtered_data.columns = filtered_data.iloc[2,:]  #Change the colnames to represent the soy flow
    clean_data = filtered_data.iloc[4:,:] #Select the useful rows
    clean_data = clean_data.replace(":", np.nan) #Make the missing values be represented as nan
    clean_data = clean_data.apply(pd.to_numeric, errors='coerce') #Make all cells be represented as numeric
    clean_data = clean_data.set_index(years)
    export_dfs[value] = clean_data #Save the df in a dictionary
    total_export = clean_data.sum(axis=1) #sum the export
    oil_export = clean_data['Oilcake & other solid residues of oil from soya beans'] #Select the different exports
    bean_export = clean_data['Soya beans'] #Select the different exports
    meal_export =clean_data['Soya bean oil and its fractions'] #Select the different exports
    export_tot[value] = total_export #Append the df
    export_beans[value] = bean_export #Append the df
    export_oil[value] = oil_export #Append the df
    export_meal[value] = meal_export #Append the df


#%%
#transpose the dfs
import_tot = import_tot.transpose()
import_beans = import_beans.transpose()
import_oil = import_oil.transpose()
import_meal = import_meal.transpose()

#transpose the dfs
export_tot = export_tot.transpose()
export_beans = export_beans.transpose()
export_oil = export_oil.transpose()
export_meal = export_meal.transpose()

#%%

#Chane nan values to 0
import_tot = import_tot.replace(np.nan, 0) #Make the missing values be represented as 0
import_beans = import_beans.replace(np.nan, 0) #Make the missing values be represented as 0
import_oil = import_oil.replace(np.nan, 0) #Make the missing values be represented as 0
import_meal = import_meal.replace(np.nan, 0) #Make the missing values be represented as 0
export_tot = export_tot.replace(np.nan, 0) #Make the missing values be represented as 0
export_beans = export_beans.replace(np.nan, 0) #Make the missing values be represented as 0
export_oil = export_oil.replace(np.nan, 0) #Make the missing values be represented as 0
export_meal = export_meal.replace(np.nan, 0) #Make the missing values be represented as 0


#%% Add some data on the continents the countries are in 
#Find the alpha 2 values
import_tot["Alpha-2 code"] = import_tot.index
import_beans["Alpha-2 code"] = import_beans.index
import_oil["Alpha-2 code"] = import_oil.index
import_meal["Alpha-2 code"] = import_meal.index
export_tot["Alpha-2 code"] = export_tot.index
export_beans["Alpha-2 code"] = export_beans.index
export_oil["Alpha-2 code"] = export_oil.index
export_meal["Alpha-2 code"] = export_meal.index

#Same for the production data
production_EU_clean["Alpha-2 code"] = production_EU_clean.index

#change the row of the GDF data of serbia so that the ISO_A2_EH cell equals XS
gdf_world.loc[gdf_world["NAME_EN"] == "Serbia", "ISO_A2_EH"] = "XS"

#Only keep the interesting columns from the gdf_world dataset
columns_to_subset = ["ISO_A2_EH", "NAME_EN", "REGION_UN", "ISO_A3_EH"]
world_geometries = gdf_world[columns_to_subset]

#Merge with the polygon layer to get the countries in the dataframe
import_tot = pd.merge(import_tot, world_geometries, left_on="Alpha-2 code", right_on="ISO_A2_EH", how = "left")
import_beans = pd.merge(import_beans, world_geometries, left_on="Alpha-2 code", right_on="ISO_A2_EH", how = "left")
import_oil = pd.merge(import_oil, world_geometries, left_on="Alpha-2 code", right_on="ISO_A2_EH", how = "left")
import_meal = pd.merge(import_meal, world_geometries, left_on="Alpha-2 code", right_on="ISO_A2_EH", how = "left")
export_tot = pd.merge(export_tot, world_geometries, left_on="Alpha-2 code", right_on="ISO_A2_EH", how = "left")
export_beans = pd.merge(export_beans, world_geometries, left_on="Alpha-2 code", right_on="ISO_A2_EH", how = "left")
export_oil = pd.merge(export_oil, world_geometries, left_on="Alpha-2 code", right_on="ISO_A2_EH", how = "left")
export_meal = pd.merge(export_meal, world_geometries, left_on="Alpha-2 code", right_on="ISO_A2_EH", how = "left")

#Same for the production data
production_EU_countries = pd.merge(production_EU_clean, world_geometries, left_on="Alpha-2 code", right_on="ISO_A2_EH", how = "left")

#Check the rows that contain an nan value
rows_with_nan_mask = import_tot.isna().any(axis=1) #Create mask
rows_with_nan = import_tot[rows_with_nan_mask] #filter the df

#Drop the rows that contain NA values. The rows have been checked by hand to ensure that there is a minimal loss of data. Only bonaire, cueta, and mellila have been left out but trade with these countries is minimal
import_tot_clean = import_tot.dropna()
import_beans_clean = import_beans.dropna()
import_oil_clean = import_oil.dropna()
import_meal_clean = import_meal.dropna()
export_tot_clean = export_tot.dropna()
export_beans_clean = export_beans.dropna()
export_oil_clean = export_oil.dropna()
export_meal_clean = export_meal.dropna()

#Drop duplicate rows
import_tot_clean = import_tot_clean.drop_duplicates("ISO_A3_EH")
import_beans_clean = import_beans_clean.drop_duplicates("ISO_A3_EH")
import_oil_clean = import_oil_clean.drop_duplicates("ISO_A3_EH")
import_meal_clean = import_meal_clean.drop_duplicates("ISO_A3_EH")
export_tot_clean = export_tot_clean.drop_duplicates("ISO_A3_EH")
export_beans_clean = export_beans_clean.drop_duplicates("ISO_A3_EH")
export_oil_clean = export_oil_clean.drop_duplicates("ISO_A3_EH")
export_meal_clean = export_meal_clean.drop_duplicates("ISO_A3_EH")

#%% Sum the continents
import_tot_continents = import_tot_clean.groupby('REGION_UN').sum().reset_index()
import_beans_continents = import_beans_clean.groupby('REGION_UN').sum().reset_index()
import_oil_continents = import_oil_clean.groupby('REGION_UN').sum().reset_index()
import_meal_continents = import_meal_clean.groupby('REGION_UN').sum().reset_index()
export_tot_continents = export_tot_clean.groupby('REGION_UN').sum().reset_index()
export_beans_continents = export_beans_clean.groupby('REGION_UN').sum().reset_index()
export_oil_continents = export_oil_clean.groupby('REGION_UN').sum().reset_index()
export_meal_continents = export_meal_clean.groupby('REGION_UN').sum().reset_index()

#%% Make nice dataframes that can be used
#Drop continents and ISO_codes from the previous dataframes
import_tot_simple = import_tot_clean.drop(['Alpha-2 code', 'ISO_A2_EH', 'REGION_UN'], axis=1)
import_beans_simple = import_beans_clean.drop(['Alpha-2 code', 'ISO_A2_EH', 'REGION_UN'], axis=1)
import_oil_simple = import_oil_clean.drop(['Alpha-2 code', 'ISO_A2_EH', 'REGION_UN'], axis=1)
import_meal_simple = import_meal_clean.drop(['Alpha-2 code', 'ISO_A2_EH', 'REGION_UN'], axis=1)
export_tot_simple = export_tot_clean.drop(['Alpha-2 code', 'ISO_A2_EH', 'REGION_UN'], axis=1)
export_beans_simple = export_beans_clean.drop(['Alpha-2 code', 'ISO_A2_EH', 'REGION_UN'], axis=1)
export_oil_simple = export_oil_clean.drop(['Alpha-2 code', 'ISO_A2_EH', 'REGION_UN'], axis=1)
export_meal_simple = export_meal_clean.drop(['Alpha-2 code', 'ISO_A2_EH', 'REGION_UN'], axis=1)

#Production
production_EU_simple = production_EU_countries.drop(['Alpha-2 code', 'ISO_A2_EH', 'REGION_UN'], axis=1)

#Change the format of the dataframes
import_tot_melted = pd.melt(import_tot_simple, id_vars=['NAME_EN', 'ISO_A3_EH'], var_name='Year', value_name='Value')
import_beans_melted = pd.melt(import_beans_simple, id_vars=['NAME_EN', 'ISO_A3_EH'], var_name='Year', value_name='Value')
import_oil_melted = pd.melt(import_oil_simple, id_vars=['NAME_EN', 'ISO_A3_EH'], var_name='Year', value_name='Value')
import_meal_melted = pd.melt(import_meal_simple, id_vars=['NAME_EN', 'ISO_A3_EH'], var_name='Year', value_name='Value')
export_tot_melted = pd.melt(export_tot_simple, id_vars=['NAME_EN', 'ISO_A3_EH'], var_name='Year', value_name='Value')
export_beans_melted = pd.melt(export_beans_simple, id_vars=['NAME_EN', 'ISO_A3_EH'], var_name='Year', value_name='Value')
export_oil_melted = pd.melt(export_oil_simple, id_vars=['NAME_EN', 'ISO_A3_EH'], var_name='Year', value_name='Value')
export_meal_melted = pd.melt(export_meal_simple, id_vars=['NAME_EN', 'ISO_A3_EH'], var_name='Year', value_name='Value')
production_EU_melted = pd.melt(production_EU_simple, id_vars=['NAME_EN', 'ISO_A3_EH'], var_name='Year', value_name='Value')

#Sort the dfs
import_tot_melted = import_tot_melted.sort_values(by=['NAME_EN', 'Year']).reset_index(drop=True)
import_beans_melted = import_beans_melted.sort_values(by=['NAME_EN', 'Year']).reset_index(drop=True)
import_oil_melted = import_oil_melted.sort_values(by=['NAME_EN', 'Year']).reset_index(drop=True)
import_meal_melted = import_meal_melted.sort_values(by=['NAME_EN', 'Year']).reset_index(drop=True)
export_tot_melted = export_tot_melted.sort_values(by=['NAME_EN', 'Year']).reset_index(drop=True)
export_beans_melted = export_beans_melted.sort_values(by=['NAME_EN', 'Year']).reset_index(drop=True)
export_oil_melted = export_oil_melted.sort_values(by=['NAME_EN', 'Year']).reset_index(drop=True)
export_meal_melted = export_meal_melted.sort_values(by=['NAME_EN', 'Year']).reset_index(drop=True)

#%%
import_tot_melted = import_tot_melted.sort_values(by=['Year', 'Value'], ascending=[False, False]).reset_index(drop=True)

names_to_in_exclude = ["USA", "BRA", "ARG"]
top3_import_total = import_tot_melted[import_tot_melted['ISO_A3_EH'].isin(names_to_in_exclude)]

other_imports = import_tot_melted[~import_tot_melted['ISO_A3_EH'].isin(names_to_in_exclude)]
other_import_total = other_imports.groupby("Year", as_index=False).sum()
other_import_total["ISO_A3_EH"] = 'Other'
other_import_total["NAME_EN"] = 'Other'

top3_import_total_other = pd.concat([top3_import_total, other_import_total], axis=0, ignore_index=True)
top3_import_total_other = top3_import_total_other.sort_values(by=['Year', 'Value'], ascending=[True, False]).reset_index(drop=True)

#%%
import_tot_melted = import_tot_melted.sort_values(by=['NAME_EN', 'Year']).reset_index(drop=True)
import_tot_melted["Year"] = pd.to_numeric(import_tot_melted["Year"], errors="coerce").astype(int)

import_tot_melted = import_tot_melted.sort_values(by=['NAME_EN', 'Year']).reset_index(drop=True)
import_beans_melted["Year"] = pd.to_numeric(import_beans_melted["Year"], errors="coerce").astype(int)
import_meal_melted["Year"] = pd.to_numeric(import_meal_melted["Year"], errors="coerce").astype(int)
import_oil_melted["Year"] = pd.to_numeric(import_oil_melted["Year"], errors="coerce").astype(int)
import_tot_melted["Year"] = pd.to_numeric(import_tot_melted["Year"], errors="coerce").astype(int)
export_tot_melted["Year"] = pd.to_numeric(export_tot_melted["Year"], errors="coerce").astype(int)
export_beans_melted["Year"] = pd.to_numeric(export_beans_melted["Year"], errors="coerce").astype(int)
export_oil_melted["Year"] = pd.to_numeric(export_oil_melted["Year"], errors="coerce").astype(int)
export_meal_melted["Year"] = pd.to_numeric(export_meal_melted["Year"], errors="coerce").astype(int)


app = Dash(__name__)

server = app.server
# Sort and process your dataframes

app.layout = dbc.Container([
    html.H1('Soy Trade with the EU', style={'textAlign': 'center'}),
    
    html.P("Figures for the trade of soy to and from the EU. Import shows where the soy that is imported to the EU is from, and export shows where the soy that is exported from the EU goes to. Different soy products can be selected, and below the graph is a slider for year selection."),

    
    html.Div(
        [dcc.Dropdown(
                options=[
                    {'label': 'Soybeans', 'value': 'soybeans'},
                    {'label': 'Soymeal', 'value': 'soymeal'},
                    {'label': 'Soy oils', 'value': 'soyoils'},
                    {'label': 'Total', 'value': 'total'}
                ],
                value='total',
                id='product-dropdown',
                style={'margin-bottom': '10px', 'width': '50%'}
            ),
        dcc.RadioItems(
            options=[
                {'label': 'Imports', 'value': 'imports'},
                {'label': 'Exports', 'value': 'exports'}
            ],
            value='imports',
            id='trade-type-radio'
        ),
    ], style={'display': 'flex'}),
    
    dcc.Tabs(
        id='tabs',
        value='tab-1',
        children=[
            dcc.Tab(label='Map', value='tab-1'),
            dcc.Tab(label='Bar chart', value='tab-2')
        ]
    ),

    dcc.Loading(
        id="loading",
        type="default",  # or "circle"
        children=[
            # This is where your graph or other content goes
            html.Div(id='tabs-content'),
        ]
    ),

    # Year slider
    dcc.Slider(
        id='year-slider',
        min=import_tot_melted['Year'].min(),
        max=import_tot_melted['Year'].max(),
        step=1,
        marks={str(year): str(year) for year in import_tot_melted['Year'].unique()},
        value=import_tot_melted['Year'].min()  # Set initial value to the maximum year
    ),
], fluid=True)

@app.callback(
    Output('tabs-content', 'children'),
    [Input('product-dropdown', 'value'),
     Input('trade-type-radio', 'value'),
     Input('year-slider', 'value'),
     Input('tabs', 'value')]
)
def update_graph(selected_product, selected_trade_type, selected_year, selected_tab):
    
    if selected_product == 'soybeans':
        if selected_trade_type == 'imports':
            df = import_beans_melted 
        else: 
            df = export_beans_melted
    elif selected_product == 'soymeal':
        if selected_trade_type == 'imports':
            df = import_meal_melted 
        else:
            df = export_meal_melted
    elif selected_product == 'soyoils':
        if selected_trade_type == 'imports':
            df = import_oil_melted
        else:
            df = export_oil_melted
    elif selected_product == 'total':
         if selected_trade_type == 'imports':
             df = import_tot_melted
         else:
             df = export_tot_melted
    else:
        # Default empty DataFrame if an unknown product is selected
        df = pd.DataFrame()
    
    df_filtered = df[df['Year'] == selected_year]

    fig = px.scatter_geo(
        df_filtered,
        locations="ISO_A3_EH",
        hover_name="NAME_EN",
        size="Value",
        animation_frame="Year",
        projection="natural earth",
        size_max=30,
        hover_data={"NAME_EN": False, "Value": True, "ISO_A3_EH": False, "Year": False }
    )
        
    # Additional callback logic for the second graph
    if selected_tab == 'tab-2':
        fig = px.bar(
            df_filtered.sort_values(by="Value", ascending=False).head(7),
            x="NAME_EN",
            y="Value"
        )
    
    return [dcc.Graph(figure=fig)]


if __name__ == '__main__':
    app.run(jupyter_mode="external", port = 8051)

