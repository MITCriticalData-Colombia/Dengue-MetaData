import pandas as pd
import numpy as np
import jellyfish as jf
import os
import sys
if not sys.warnoptions:
    import warnings
    warnings.simplefilter("ignore")
from functions import function1, function2, normalize, size_error
#####################################################################################################
# Global variables
#####################################################################################################

columns_names = ['Municipality code',
                'Age 0-4 (%)',
                'Age 5-14 (%)',
                'Age 15-29 (%)',
                'Age >30 (%)',
                'Afrocolombian Population (%)',
                'Indian Population (%)',
                'People with Disabilities (%)',
                'People who cannot read or write (%)',
                'Secondary/Higher Education (%)',
                'Employed population (%)',
                'Unemployed population (%)',
                'People doing housework (%)',
                'Retired people (%)',
                'Men (%)',
                'Women (%)',
                'Households without water access (%)',
                'Households without internet access (%)',
                'Building stratification 1 (%)',
                'Building stratification 2 (%)',
                'Building stratification 3 (%)',
                'Building stratification 4 (%)',
                'Building stratification 5 (%)',
                'Building stratification 6 (%)',
                'Number of hospitals per Km2',
                'Number of houses per Km2']

#####################################################################################################
# Paths of Files
#####################################################################################################

# Static Paths
main_path = "D:/UNIVERSIDAD/DANE Dengue/BasesDatos"
dengue_data_file = "Data_Files/DANE_Dengue_Data_2015_2019.csv"
municipality_area_file = "Data_Files/Municipality_Area.csv"
# Reading the static .csv files
municipality_data = pd.read_csv(dengue_data_file, usecols=['State code','Municipality code', 'Municipality'])
municipality_area_data = pd.read_csv(municipality_area_file)
# Mayusculas y quito tildes
for i in range(len(municipality_area_data['Departamento'])):
    municipality_area_data.loc[i,'Departamento'] = normalize(municipality_area_data.loc[i,'Departamento'].upper())
main_file = pd.read_csv(dengue_data_file)
municipalities_df = []
# List of main directories
states = os.listdir(main_path)

for i in range(len(states)):
    # Dynamic Paths
    people_file_path = main_path + '/' + states[i] + '/CNPV2018_5PER_A2_' + states[i][0:2] + '.csv'
    houses_file_path = main_path + '/' + states[i] + '/CNPV2018_2HOG_A2_' + states[i][0:2] + '.csv'
    viv_file_path = main_path + '/' + states[i] + '/CNPV2018_1VIV_A2_' + states[i][0:2] + '.csv'
    health_providers_file = main_path + '/' + states[i] + '/Prestadores_' + states[i] + '.csv'
    # Reading the dynamic .csv files
    people_data = pd.read_csv(people_file_path, usecols=['U_MPIO', 'P_EDADR', 'PA1_GRP_ETNIC', 'CONDICION_FISICA', 'P_ALFABETA', 'P_NIVEL_ANOSR', 'P_TRABAJO', 'P_SEXO'])
    houses_data = pd.read_csv(houses_file_path, usecols=['U_MPIO','COD_ENCUESTAS'])
    viv_data = pd.read_csv(viv_file_path, usecols=['U_MPIO', 'VA1_ESTRATO', 'VB_ACU', 'VF_INTERNET'])
    health_providers_data = pd.read_csv(health_providers_file, usecols=['depa_nombre', 'muni_nombre','nombre_prestador'])

    #####################################################################################################
    # Inicio del código
    #####################################################################################################

    state_code = int(states[i][0:2])
    state_name = normalize(states[i][2:].upper())

    #####################################################################################################
    # Se van a encontrar las variables posibles en el archivo de 'Personas' y 'Viviendas'
    #####################################################################################################

    # s será la matriz de salida de tamaño (N°deMunicipios,26)
    s = function1(people_data, viv_data)

    #######################################################################################
    # Ahora se va a calcular el numero de hospitales por km2 y el numero de hogares por km2
    #######################################################################################

    s[:,24:26] = function2(state_code, state_name, municipality_data, health_providers_data, municipality_area_data, houses_data)
    # Redondeamos todos los resultados a dos decimales y se escriben como porcentaje, menos las ultimas dos filas
    s[:,1:-8] = np.round(s[:,1:-8]*100,2)
    s[:,-8:-2] = np.round(s[:,-8:-2]*100,4)
    s[:,-2:] = np.round(s[:,-2:],6)
    # Se ajustan los codigos de cada municipio para el merge final
    s[:,0] = s[:,0] + state_code*1000

    #####################################################################################################
    # Merging with the main file
    #####################################################################################################

    Municipality_Data = pd.DataFrame(s, columns = columns_names)
    municipalities_df.append(Municipality_Data)

# Optimization
final = municipalities_df[0]
for i in range(1,len(municipalities_df)):
    final = final.append(municipalities_df[i],ignore_index=True)
print(final)
# Falta eliminar columnas innecesarias antes del merge
main_file = main_file.merge(final, on='Municipality code', how='outer')
main_file['Municipality code'] = main_file['Municipality code'].apply(int)
main_file['Municipality code'] = pd.to_numeric(main_file['Municipality code'], downcast='integer')
print(main_file)
size_error(main_file,1122)
# Creating the csv file
main_file.to_csv('DANE_Dengue_Data_Variables.csv', index=False)
