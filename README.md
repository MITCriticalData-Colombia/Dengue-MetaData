# Dengue-MetaData

## Introduction
This branch contains the csv files that describes the socioeconomic, demographic and some meteorological data such as temperature and precipitation that describe aspects of the Colombian population such as lifestyle and geography.

## Files
<ul>
  <li><h3>Data_Files:<h3></li> This folder contains the base datasets for the construction of the final dataset, has four csv files:
        <ul>
          <li><h4>DANE_Dengue_Data_2015_2019.csv :</h4></li> In this file you'll find the sociodemograhic, and dengue cases data
          <li><h4>Municipality_Area.csv :</h4></li> This file has the area in square km in each municipality
          <li><h4>worldclim_precipitation_2015-2018.csv :</h4></li> This file has the data of the mean precipitation of each municipality each month of the year from 2015 till 2018, and the total precipitation each year.
          <li><h4>worldclim_temperature_2015-2018.csv :</h4></li> This file has the data of the mean temperature of each municipality each month of the year from 2015 till 2018, and the average temperature each year.
          <li><h4>tempearture2007-2018.csv :</h4></li> This file has the data of the averqge temperature of each municipality each month of the year from 2007 till 2018, and the average temperature each year.
        </ul>
  
  <li><h3>WorldClimTemperature2007_2018.ipynb :</h3></li> This file contains a Google Collab notebook, to extract the average temperature values for each Municipality of Colombia in each month from 2007 till 2018
  
  <li><h3>DANE_Dengue_Data_Variables.csv :</h3></li> Final file before add the themperature and precipitation data
        <h4> Variables </h4>
        <ul>
          <li>Cases:</li> This variable describes the number of dengue cases between 2018 and 2020.
          <li>Ages (%): </li> This variable describes the age range of the population.
          <li>Afrocolombian Population (%):</li>
          <li>Indian Population (%): </li>
          <li>People with Disabilities (%):</li> This variable describes the group of people who have some physical, psychological or mental limitation.
          <li>People who cannot read or write (%): </li> This variable describes the group of people who have not been able to develop skills such as reading or writing.
          <li>Secondary/Higher Education (%):</li> This variable describes the group of people who are in school or academic process.
          <li>Employed population (%):</li> This variable describes the group of people who have not been able to develop skills such as reading or writing.
          <li>Unemployed population (%):</li> This variable describes the group of people who are not engaged in work.
          <li>People doing housework (%):</li> This variable describes the group of people who perform household chores.
          <li>Retired people (%):</li> This variable describes the group of people who have completed their work cycle.
          <li>Gender (%):</li> This variable describes the classification between women and men.
          <li>Households without water access (%):</li> This variable describes the group of people who have access to water service at home.
          <li>Households without internet access (%): </li> This variable describes the group of people who have internet service at home.
          <li>Building stratification (%):</li> This variable describes the groups in which the population is divided according to parameters, which could be location, economic level, quality of services received, among others.
          <li>Number of hospitals per Km2:</li> This variable describes the number of hospitals per Km2.
          <li>Number of houses per Km2: </li> This variable describes the number of households per Km2.
        </ul>

  
  <li><h3> dengue_temperature_precipitation_2015-2019.csv :</h3></li> This is the final file, with sociodemographic data, temperature, precipitations, and dengue cases, this file contains <strong>The same variables of DANE_Dengue_Data_Variables.csv and other extra variables:</strong>
  <h4> Extra variables: </h4>
        <ul>
          <li>Precipitation by month:</li> These variables have the aveage precipitation each month in a specific municipality follow a pattern: <strong> PRECIPITATION_Month_year </strong>
          <li>Precipitation by year:</li> These variables are the sum of total precipitation in a year and follow a pattern: <strong> PerPAnual_year </strong>
          <li>Temperature by month:</li> These variables have the aveage temperature each month in a specific municipality follow a pattern: <strong> TEMPERATURE_Month_year </strong>
          <li>Temperature by year:</li> These variables are the average of temperature in a year and follow a pattern: <strong> TMAnual_year </strong>
        </ul>  

</ul>
