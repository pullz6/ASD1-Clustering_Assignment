# -*- coding: utf-8 -*-
"""
Created on Sun Dec 12 08:57:02 2022

@author: Pulsara
"""

#Importing modules

import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt 
import scipy.optimize as opt
import sklearn.cluster as cluster
import sklearn.metrics as skmet
import itertools as iter

#Defining the functions to be used

def return_Dataframe_two(filename):
    """"This function returns the two dataframes from the entire dataset, one with years as columns and another with countries as columns"""
    #Reading the CSV file and adding it to a dataframe, this dataframe will have the years as columns
    df = pd.read_csv(filename)
    #Creating the transposed dataframe with countries as columns using a function, we are also passing the required parameters
    df_t = return_transposed_df(df,4,0)
    #Returning the dataframes
    return df,df_t
    
def return_dataframes(df, Indicator):
    """"This function returns the dataframe by filtering a specific indicator or a factor"""
    #If statement to filter the factor
    df_temp = df[df['Indicator Name']== Indicator]
    #Resetting our index
    df_temp = df_temp.reset_index()
    df_temp = df_temp.replace(np.nan,0)
    #Creating the transposed dataframe with countries as columns using a function, we are also passing the required parameters
    df_temp_t = return_transposed_df(df_temp,5,1)
    #Returning the dataframes
    return df_temp, df_temp_t


def return_transposed_df(df, i, j):
    """"This function returns the transposed dataframe of a passed dataframe along with index and header parameters"""
    #Transposing the passed datafram
    df_t = pd.DataFrame.transpose(df)
    #Selecting the header column
    header = df_t.iloc[j].values.tolist()
    #Including the header column
    df_t.columns = header
    #Dropping transposed rows that are not neccessary for our calculations
    df_t = df_t.drop(df_t.index[0:i])
    #Resetting our index
    df_t = df_t.reset_index()
    df_t = df_t.replace(np.nan,0)
    #Renaming the year column
    df_t.columns.values[0] = "Year"
    return df_t

def return_year_df(year, df):
    """"This function returns a list for values for a specific factor for particular year and for 5 countries"""
    #Create empty list
    year_list = []
    #Select all the values of that factor in one year
    df_temp = df[df["Year"]==year]
    #Include the values for the selected countries in the list
    year_list.append(df_temp.iloc[0]['United States'])
    year_list.append(df_temp.iloc[0]['China'])
    year_list.append(df_temp.iloc[0]['United Kingdom'])
    year_list.append(df_temp.iloc[0]['India'])
    year_list.append(df_temp.iloc[0]['Saudi Arabia'])
    #Return the list 
    return year_list

def countrywise_dataframe (country, df_pop_total, df_Co2_liq, df_Co2_solid, df_agri_land, df_urban_population, df_Co2_total, df_forest_area, df_arable_lands): 
    """"This function returns a dataframe consisting of all the yearly values for each selected factor related to a particular country"""
    #Create empty dataframe       
    df = pd.DataFrame()
    #If statement to filter if the particular country and add it to the new dataframe under accurate column names. 
    if country == "China" : 
        df['Population'] = df_pop_total['China']
        df['CO2_Liquid'] = df_Co2_liq['China']
        df['CO2_Solid'] = df_Co2_solid['China']
        df['Agri_land'] = df_agri_land['China']
        df['Co2_Total'] = df_Co2_total['China']
        df['Urban Population'] = df_urban_population['China']
        df['Forest Area'] = df_forest_area['China']
        df['Arable Lands'] = df_arable_lands['China']
    if country == "India" :
        df['Population'] = df_pop_total['India']
        df['CO2_Liquid'] = df_Co2_liq['India']
        df['CO2_Solid'] = df_Co2_solid['India']
        df['Agri_land'] = df_agri_land['India']
        df['Urban Population'] = df_urban_population['India']
        df['Co2_Total'] = df_Co2_total['India']
        df['Forest Area'] = df_forest_area['India']
        df['Arable Lands'] = df_arable_lands['India']
    if country == "Saudi Arabia" : 
        df['Population'] = df_pop_total['Saudi Arabia']
        df['CO2_Liquid'] = df_Co2_liq['Saudi Arabia']
        df['CO2_Solid'] = df_Co2_solid['Saudi Arabia']
        df['Agri_land'] = df_agri_land['Saudi Arabia']
        df['Urban Population'] = df_urban_population['Saudi Arabia']
        df['Co2_Total'] = df_Co2_total['Saudi Arabia']
        df['Forest Area'] = df_forest_area['Saudi Arabia']
        df['Arable Lands'] = df_arable_lands['Saudi Arabia']
    if country == "United States" : 
        df['Population'] = df_pop_total['United States']
        df['CO2_Liquid'] = df_Co2_liq['United States']
        df['CO2_Solid'] = df_Co2_solid['United States']
        df['Agri_land'] = df_agri_land['United States']
        df['Urban Population'] = df_urban_population['United States']
        df['Co2_Total'] = df_Co2_total['United States']
        df['Forest Area'] = df_forest_area['United States']
        df['Arable Lands'] = df_arable_lands['United States']
    if country == "United Kingdom" : 
        df['Population'] = df_pop_total['United Kingdom']
        df['CO2_Liquid'] = df_Co2_liq['United Kingdom']
        df['CO2_Solid'] = df_Co2_solid['United Kingdom']
        df['Agri_land'] = df_agri_land['United Kingdom']
        df['Urban Population'] = df_urban_population['United Kingdom']
        df['Co2_Total'] = df_Co2_total['United Kingdom']
        df['Forest Area'] = df_forest_area['United Kingdom']
        df['Arable Lands'] = df_arable_lands['United Kingdom']
    #Delete any zero values in each of the columns
    df = df[(df[['Population','CO2_Liquid','CO2_Solid','Agri_land','Urban Population']] != 0).all(axis=1)]
    #Return the dataframe
    return df
    
def norm(array):
    """"eturns array normalised to [0,1]. Array can be a numpy array or a column of a dataframe"""
    min_val = np.min(array)
    max_val = np.max(array)
    scaled = (array-min_val) / (max_val-min_val)
    return scaled

def norm_df(df, first=1, last=None):
    """ Returns all columns of the dataframe normalised to [0,1] with the exception of the first (containing the names) Calls function norm to do the normalisation of one column, but doing all in one function is also fine. First, last: columns from first to last (including) are normalised. Defaulted to all. None is the empty entry. The default corresponds"""
    # iterate over all numerical columns
    for col in df.columns[first:last]: # excluding the first column
        df[col] = norm(df[col])
    df = df.replace(np.nan,0)
    return df

def cleaning_df(df): 
    """"This function returns a cleaned dataframe without the unwanted columns"""
    df = df.drop(['Country Name','Country Code','Indicator Name','Indicator Code'], axis=1)
    return df 

def exponential(t, n0, g):
    t = t - 1960.0
    f = n0 * np.exp(g*t)
    return f

#Main Programme

df, df_t = return_Dataframe_two('API_19_DS2_en_csv_v2_4700503.csv')

#Calling the function to get two dataframes for particular factor or climate indicator
df_population, df_population_t = return_dataframes(df, "Population, total")
df_pop_growth, df_pop_growth_t = return_dataframes(df, "Population growth (annual %)") 
df_Co2_liquid, df_Co2_liquid_t = return_dataframes(df, "CO2 emissions from liquid fuel consumption (kt)")
df_Co2_solid, df_Co2_solid_t = return_dataframes(df, "CO2 emissions from solid fuel consumption (kt)")
df_Co2_total, df_Co2_total_t = return_dataframes(df, "CO2 emissions (kt)")
df_Co2_gas, df_Co2_fas_t = return_dataframes(df,"CO2 emissions from gaseous fuel consumption (kt)")
df_ara_lands, df_ara_lands_t = return_dataframes(df, "Arable land (% of land area)")
df_agri_lands, df_agri_lands_t = return_dataframes(df, "Agricultural land (sq. km)")
df_urban_growth, df_urban_growth_t = return_dataframes(df, "Urban population growth (annual %)")
df_urban_pop, df_urban_pop_t = return_dataframes(df, "Urban population")
df_urban_percent, df_urban_percent_t = return_dataframes(df, "Urban population (% of total population)")
df_forest_area, df_forest_area_t = return_dataframes(df, "Forest area (sq. km)") 

#Cleaning our dataframes for better data processing
df_Co2_total = cleaning_df(df_Co2_solid)
df_population = cleaning_df(df_population)
df_forest_area = cleaning_df(df_forest_area)
df_agri_lands = cleaning_df(df_agri_lands)
df_pop_growth = cleaning_df(df_pop_growth)
df_urban_pop = cleaning_df(df_urban_pop)

#Creating the normalised version of each of the factors 
df_pop_norm = df_population.copy()
df_pop_norm = norm_df(df_pop_norm)
df_co2_tot_norm = norm_df(df_Co2_total)
df_forest_area_norm = norm_df(df_forest_area)
df_agri_lands_norm = norm_df(df_agri_lands)
df_urban_growth_norm = norm_df(df_urban_pop)

#Creating a dataframe to contain all the facotrs for all counties in 1960 so we can understand which clusters lie with
df_1960 = pd.DataFrame()
df_1960['Population'] = df_pop_norm['1960']
df_1960['Co2 Total'] = df_co2_tot_norm['1960']
df_1960['Forest Area'] = df_forest_area_norm['1960']
df_1960['Agriculatural Lands'] = df_agri_lands_norm['1960']
df_1960['Urban growth'] = df_urban_growth_norm['1960']

for ic in range(2, 7):
    # set up kmeans and fit
    kmeans = cluster.KMeans(n_clusters=ic)
    kmeans.fit(df_1960)
    # extract labels and calculate silhoutte score
    labels = kmeans.labels_
    print (ic, skmet.silhouette_score(df_1960, labels))

#Output from the silhouette score is clusters number = 2 and clusters number = 3 would be the best 
#For cluster number = 2
kmeans = cluster.KMeans(n_clusters=2)
clusters = kmeans.fit(df_1960)
# extract labels and cluster centres
labels = kmeans.labels_
df_population['Cluster'] = kmeans.labels_
cen = kmeans.cluster_centers_

plt.figure()
plt.scatter(df_1960['Population'],df_1960['Urban growth'], c=labels, s=50)
plt.scatter(cen[:, 0], cen[:, 1], s=200, c='black')
#plt.plot(5.01900000e+03, 3.30566808e-02, marker="o", markersize=5, markeredgecolor="red", markerfacecolor="green")
#plt.plot(ypoints[0], ypoints[1], marker="o", markersize=20, markeredgecolor="red", markerfacecolor="blue")
# colour map Accent selected to increase contrast between colours
plt.colorbar()
    
plt.xlabel("Population in 1960")
plt.ylabel("Urban Growth in 1960")
plt.title("2 clusters")
plt.show()

#For cluster number = 3
kmeans = cluster.KMeans(n_clusters=3)
clusters = kmeans.fit(df_1960)
# extract labels and cluster centres
labels = kmeans.labels_
df_1960['Cluster'] = kmeans.labels_
cen = kmeans.cluster_centers_

plt.figure()
plt.scatter(df_1960['Population'],df_1960['Urban growth'], c=labels, s=50)
plt.scatter(cen[:, 0], cen[:, 1], s=200, c='black')
#plt.plot(5.01900000e+03, 3.30566808e-02, marker="o", markersize=5, markeredgecolor="red", markerfacecolor="green")
#plt.plot(ypoints[0], ypoints[1], marker="o", markersize=20, markeredgecolor="red", markerfacecolor="blue")
# colour map Accent selected to increase contrast between colours
plt.colorbar()
    
plt.xlabel("Population in 1960")
plt.ylabel("Urban Growth in 1960")
plt.title("3 clusters")
plt.show()

#Creating a dataframe to contain all the facotrs for all counties in 2000 so we can understand which clusters lie with
df_2000 = pd.DataFrame()
df_2000['Population'] = df_pop_norm['2000']
df_2000['Co2 Total'] = df_co2_tot_norm['2000']
df_2000['Forest Area'] = df_forest_area_norm['2000']
df_2000['Agriculatural Lands'] = df_agri_lands_norm['2000']
df_2000['Urban growth'] = df_urban_growth_norm['2000']

for ic in range(2, 7):
    # set up kmeans and fit
    kmeans = cluster.KMeans(n_clusters=ic)
    kmeans.fit(df_2000)
    # extract labels and calculate silhoutte score
    labels = kmeans.labels_
    print (ic, skmet.silhouette_score(df_2000, labels))

#Output from the silhouette score is clusters number = 2 and clusters number = 3 would be the best 
#For cluster number = 2
kmeans = cluster.KMeans(n_clusters=2)
clusters = kmeans.fit(df_2000)
# extract labels and cluster centres
labels = kmeans.labels_
df_population['Cluster'] = kmeans.labels_
cen = kmeans.cluster_centers_

plt.figure()
plt.scatter(df_2000['Population'],df_2000['Urban growth'], c=labels, s=50)
plt.scatter(cen[:, 0], cen[:, 1], s=200, c='black')
#plt.plot(5.01900000e+03, 3.30566808e-02, marker="o", markersize=5, markeredgecolor="red", markerfacecolor="green")
#plt.plot(ypoints[0], ypoints[1], marker="o", markersize=20, markeredgecolor="red", markerfacecolor="blue")
# colour map Accent selected to increase contrast between colours
plt.colorbar()
    
plt.xlabel("Population in 2000")
plt.ylabel("Urban Growth in 2000")
plt.title("2 clusters")
plt.show()

#For cluster number = 3
kmeans = cluster.KMeans(n_clusters=3)
clusters = kmeans.fit(df_2000)
# extract labels and cluster centres
labels = kmeans.labels_
df_1960['Cluster'] = kmeans.labels_
cen = kmeans.cluster_centers_

plt.figure()
plt.scatter(df_2000['Population'],df_2000['Urban growth'], c=labels, s=50)
plt.scatter(cen[:, 0], cen[:, 1], s=200, c='black')
#plt.plot(5.01900000e+03, 3.30566808e-02, marker="o", markersize=5, markeredgecolor="red", markerfacecolor="green")
#plt.plot(ypoints[0], ypoints[1], marker="o", markersize=20, markeredgecolor="red", markerfacecolor="blue")
# colour map Accent selected to increase contrast between colours
plt.colorbar()
    
plt.xlabel("Population in 2000")
plt.ylabel("Urban Growth in 2000")
plt.title("3 clusters")
plt.show()

# =============================================================================
# Creating a simple model with curve fit 
#Selecting the unites states dataset
df_usa = countrywise_dataframe("United States", df_population_t, df_Co2_liquid_t, df_Co2_solid_t, df_agri_lands_t, df_urban_pop_t, df_Co2_total_t, df_forest_area_t, df_ara_lands_t)
#Adding the year into the dataframe
df_usa['Year'] = df_population_t['Year']
#Creating a new data column for the relationship between population and arable lands
df_usa['Pop per arable lands'] = df_usa['Population']/df_usa['Arable Lands']
#Dropping all the zero values
df_usa = df_usa[(df_usa[['Year','Pop per arable lands']] != 0).all(axis=1)]
#Converting the selected data columns in integer since np.exp expects integers
df_usa = df_usa.astype({"Year":"int","Pop per arable lands":"int"})
#
popt, covar = opt.curve_fit(exponential, df_usa["Year"], df_usa["Pop per arable lands"]) 
sigma_exp = np.sqrt(np.diag(covar))
df_usa["pop_exp"] = exponential(df_usa["Year"], *popt)
plt.figure()
plt.plot(df_usa["Year"], df_usa["Pop per arable lands"], label="data")
plt.plot(df_usa["Year"], df_usa["pop_exp"], label="fit")
plt.legend()
plt.title("First fit attempt")
plt.xlabel("Year")
plt.ylabel("Population per Arable Land")
plt.show()
popt = [9314169.966403, 0.01]
df_usa["pop_exp"] = exponential(df_usa["Year"], *popt)
plt.figure()
plt.plot(df_usa["Year"], df_usa["Pop per arable lands"], label="data")
plt.plot(df_usa["Year"], df_usa["pop_exp"], label="fit")
plt.legend()
plt.xlabel("year")
plt.ylabel("Population per Arable Land")
plt.title("Improved start value")
plt.show()

popt, covar = opt.curve_fit(exponential, df_usa["Year"], df_usa["Pop per arable lands"], p0=[9314169.966403, 0.01]) 
df_usa["pop_exp"] = exponential(df_usa["Year"], *popt)
plt.figure()
plt.plot(df_usa["Year"], df_usa["Pop per arable lands"], label="data")
plt.plot(df_usa["Year"], df_usa["pop_exp"], label="fit")
plt.legend()
plt.title("Final fit attempt")
plt.xlabel("Year")
plt.ylabel("Population per Arable Land")
plt.show()

years_to_pred = np.arange(1961, 2033)
pred_exp = exponential(years_to_pred, *popt)

#Calculating the upper and lower ranges 
#lower_limit, upper_limit = err_ranges(years_to_pred, exponential, popt, sigma_exp)
#def err_ranges(x, func, param, sigma)
lower = exponential(df_usa["Year"], *popt)
upper = lower
uplow = []   # list to hold upper and lower limits for parameters
zipped = zip(popt, sigma_exp)
for p,s in zipped:
    pmin = p - s
    pmax = p + s
    uplow.append((pmin, pmax))

pmix = list(iter.product(*uplow))
    
for p in pmix:
    y = exponential(df_usa["Year"], *p)
    lower = np.minimum(lower, y)
    upper = np.maximum(upper, y)


plt.figure()
plt.plot(df_usa["Year"], df_usa["Pop per arable lands"], label="Data")
plt.plot(years_to_pred, pred_exp, label="Predicted Line" , color = "yellow")
plt.fill_between(df_usa["Year"], lower, upper, color="red", alpha=0.4, label = "Confidence Range")
plt.legend()
plt.xlabel("year")
plt.ylabel("Population per Arable Land")
plt.title("Improved start value")
plt.show()

# =============================================================================



