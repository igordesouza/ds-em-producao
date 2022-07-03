# # Versão 01 - Meigarom
# import pickle
# import inflection
# import pandas as pd
# import numpy as np
# import math
# import datetime

# class Rossmann( object ):
#     def __init__( self ):
#         # self.home_path='/Users/meigarom/repos/DataScience_Em_Producao/'
#         self.home_path='/Users/igor/repos/ds-em-producao/'
#         self.competition_distance_scaler   = pickle.load( open( self.home_path + 'parameter/competition_distance_scaler.pkl', 'rb') )
#         self.competition_time_month_scaler = pickle.load( open( self.home_path + 'parameter/competition_time_month_scaler.pkl', 'rb') )
#         self.promo_time_week_scaler        = pickle.load( open( self.home_path + 'parameter/promo_time_week_scaler.pkl', 'rb') )
#         self.year_scaler                   = pickle.load( open( self.home_path + 'parameter/year_scaler.pkl', 'rb') )
#         self.store_type_scaler             = pickle.load( open( self.home_path + 'parameter/store_type_scaler.pkl', 'rb') )
        
        
#     def data_cleaning( self, df1 ): 
        
#         ## 1.1. Rename Columns
#         cols_old = ['Store', 'DayOfWeek', 'Date', 'Open', 'Promo', 'StateHoliday', 'SchoolHoliday', 
#                     'StoreType', 'Assortment', 'CompetitionDistance', 'CompetitionOpenSinceMonth',
#                     'CompetitionOpenSinceYear', 'Promo2', 'Promo2SinceWeek', 'Promo2SinceYear', 'PromoInterval']

#         snakecase = lambda x: inflection.underscore( x )

#         cols_new = list( map( snakecase, cols_old ) )

#         # rename
#         df1.columns = cols_new

#         ## 1.3. Data Types
#         df1['date'] = pd.to_datetime( df1['date'] )

#         ## 1.5. Fillout NA
#         #competition_distance        
#         df1['competition_distance'] = df1['competition_distance'].apply( lambda x: 200000.0 if math.isnan( x ) else x )

#         #competition_open_since_month
#         df1['competition_open_since_month'] = df1.apply( lambda x: x['date'].month if math.isnan( x['competition_open_since_month'] ) else x['competition_open_since_month'], axis=1 )

#         #competition_open_since_year 
#         df1['competition_open_since_year'] = df1.apply( lambda x: x['date'].year if math.isnan( x['competition_open_since_year'] ) else x['competition_open_since_year'], axis=1 )

#         #promo2_since_week           
#         df1['promo2_since_week'] = df1.apply( lambda x: x['date'].week if math.isnan( x['promo2_since_week'] ) else x['promo2_since_week'], axis=1 )

#         #promo2_since_year           
#         df1['promo2_since_year'] = df1.apply( lambda x: x['date'].year if math.isnan( x['promo2_since_year'] ) else x['promo2_since_year'], axis=1 )

#         #promo_interval              
#         month_map = {1: 'Jan',  2: 'Fev',  3: 'Mar',  4: 'Apr',  5: 'May',  6: 'Jun',  7: 'Jul',  8: 'Aug',  9: 'Sep',  10: 'Oct', 11: 'Nov', 12: 'Dec'}

#         df1['promo_interval'].fillna(0, inplace=True )

#         df1['month_map'] = df1['date'].dt.month.map( month_map )

#         df1['is_promo'] = df1[['promo_interval', 'month_map']].apply( lambda x: 0 if x['promo_interval'] == 0 else 1 if x['month_map'] in x['promo_interval'].split( ',' ) else 0, axis=1 )

#         ## 1.6. Change Data Types
#         # competiton
#         df1['competition_open_since_month'] = df1['competition_open_since_month'].astype( int )
#         df1['competition_open_since_year'] = df1['competition_open_since_year'].astype( int )

#         # promo2
#         df1['promo2_since_week'] = df1['promo2_since_week'].astype( int )
#         df1['promo2_since_year'] = df1['promo2_since_year'].astype( int )
        
#         return df1 


#     def feature_engineering( self, df2 ):

#         # year
#         df2['year'] = df2['date'].dt.year

#         # month
#         df2['month'] = df2['date'].dt.month

#         # day
#         df2['day'] = df2['date'].dt.day

#         # week of year
#         df2['week_of_year'] = df2['date'].dt.weekofyear

#         # year week
#         df2['year_week'] = df2['date'].dt.strftime( '%Y-%W' )

#         # competition since
#         df2['competition_since'] = df2.apply( lambda x: datetime.datetime( year=x['competition_open_since_year'], month=x['competition_open_since_month'],day=1 ), axis=1 )
#         df2['competition_time_month'] = ( ( df2['date'] - df2['competition_since'] )/30 ).apply( lambda x: x.days ).astype( int )

#         # promo since
#         df2['promo_since'] = df2['promo2_since_year'].astype( str ) + '-' + df2['promo2_since_week'].astype( str )
#         df2['promo_since'] = df2['promo_since'].apply( lambda x: datetime.datetime.strptime( x + '-1', '%Y-%W-%w' ) - datetime.timedelta( days=7 ) )
#         df2['promo_time_week'] = ( ( df2['date'] - df2['promo_since'] )/7 ).apply( lambda x: x.days ).astype( int )

#         # assortment
#         df2['assortment'] = df2['assortment'].apply( lambda x: 'basic' if x == 'a' else 'extra' if x == 'b' else 'extended' )

#         # state holiday
#         df2['state_holiday'] = df2['state_holiday'].apply( lambda x: 'public_holiday' if x == 'a' else 'easter_holiday' if x == 'b' else 'christmas' if x == 'c' else 'regular_day' )

#         # 3.0. PASSO 03 - FILTRAGEM DE VARIÁVEIS
#         ## 3.1. Filtragem das Linhas
#         df2 = df2[df2['open'] != 0]

#         ## 3.2. Selecao das Colunas
#         cols_drop = ['open', 'promo_interval', 'month_map']
#         df2 = df2.drop( cols_drop, axis=1 )
        
#         return df2


# #    "Dentro do método data_preparation(self, df5) não deveria ser utilizado o método transform nos scalers ao invés de fit_transform? 
# # Se for utilizado o fit_transform os scalers serão ajustados novamente do 0, perdendo o sentido de carregá-los utilizando a biblioteca pickle." CORRIGIDO

#     def data_preparation( self, df5 ):

#         ## 5.2. Rescaling 
#         # competition distance
#         df5['competition_distance'] = self.competition_distance_scaler.transform( df5[['competition_distance']].values )
    
#         # competition time month
#         df5['competition_time_month'] = self.competition_time_month_scaler.transform( df5[['competition_time_month']].values )

#         # promo time week
#         df5['promo_time_week'] = self.promo_time_week_scaler.transform( df5[['promo_time_week']].values )
        
#         # year
#         df5['year'] = self.year_scaler.transform( df5[['year']].values )

#         ### 5.3.1. Encoding
#         # state_holiday - One Hot Encoding
#         df5 = pd.get_dummies( df5, prefix=['state_holiday'], columns=['state_holiday'] )

#         # store_type - Label Encoding
#         df5['store_type'] = self.store_type_scaler.fit_transform( df5['store_type'] )

#         # assortment - Ordinal Encoding
#         assortment_dict = {'basic': 1,  'extra': 2, 'extended': 3}
#         df5['assortment'] = df5['assortment'].map( assortment_dict )

        
#         ### 5.3.3. Nature Transformation
#         # day of week
#         df5['day_of_week_sin'] = df5['day_of_week'].apply( lambda x: np.sin( x * ( 2. * np.pi/7 ) ) )
#         df5['day_of_week_cos'] = df5['day_of_week'].apply( lambda x: np.cos( x * ( 2. * np.pi/7 ) ) )

#         # month
#         df5['month_sin'] = df5['month'].apply( lambda x: np.sin( x * ( 2. * np.pi/12 ) ) )
#         df5['month_cos'] = df5['month'].apply( lambda x: np.cos( x * ( 2. * np.pi/12 ) ) )

#         # day 
#         df5['day_sin'] = df5['day'].apply( lambda x: np.sin( x * ( 2. * np.pi/30 ) ) )
#         df5['day_cos'] = df5['day'].apply( lambda x: np.cos( x * ( 2. * np.pi/30 ) ) )

#         # week of year
#         df5['week_of_year_sin'] = df5['week_of_year'].apply( lambda x: np.sin( x * ( 2. * np.pi/52 ) ) )
#         df5['week_of_year_cos'] = df5['week_of_year'].apply( lambda x: np.cos( x * ( 2. * np.pi/52 ) ) )
        
        
#         cols_selected = [ 'store', 'promo', 'store_type', 'assortment', 'competition_distance', 'competition_open_since_month',
#             'competition_open_since_year', 'promo2', 'promo2_since_week', 'promo2_since_year', 'competition_time_month', 'promo_time_week', 'day_of_week_sin', 'day_of_week_cos', 'month_sin', 'month_cos', 'day_sin', 'day_cos', 'week_of_year_sin', 'week_of_year_cos']
        
#         return df5[ cols_selected ]
    
    
#     def get_prediction( self, model, original_data, test_data ):
#         # prediction
#         pred = model.predict( test_data )
        
#         # join pred into the original data
#         original_data['prediction'] = np.expm1( pred )
        
#         return original_data.to_json( orient='records', date_format='iso' )

#==========================================================================================================

# Versão 02 - Atualizado: 02/07/22

# import pickle 
# import inflection
# import pandas as pd
# import numpy as np
# import math

# class Rossmann(object):
#     def __init__(self):
#         self.home_path='/Users/igor/repos/ds-em-producao/'
#         self.competition_distance_scaler    = pickle.load(  self.home_path + open( 'parameter/competition_distance_scaler.pkl', 'rb') )
#         self.competition_time_month_scaler  = pickle.load(  self.home_path + open( 'parameter/competition_time_month_scaler.pkl', 'rb') )
#         self.promo_time_week_scaler         = pickle.load(  self.home_path + open( 'parameter/promo_time_week_scaler.pkl', 'rb') )
#         self.year_scaler                    = pickle.load(  self.home_path + open( 'parameter/year_scaler.pkl', 'rb') )
#         self.store_type_scaler              = pickle.load(  self.home_path + open( 'parameter/store_type_scaler.pkl', 'rb') )

    
#     def data_cleaning(self,df1):
        
#         ## 1.1. Rename Columns

#         cols_old = ['Store', 'DayOfWeek', 'Date', 'Open', 'Promo', 'StateHoliday', 'SchoolHoliday', 
#                     'StoreType', 'Assortment', 'CompetitionDistance', 'CompetitionOpenSinceMonth',
#                     'CompetitionOpenSinceYear', 'Promo2', 'Promo2SinceWeek', 'Promo2SinceYear', 'PromoInterval']

#         snakecase = lambda x: inflection.underscore( x )

#         cols_new = list( map( snakecase, cols_old ) )

#         # rename
#         df1.columns = cols_new


#         ## 1.3. Data Types

#         df1['date'] = pd.to_datetime( df1['date'] )
        

#         ## 1.5. Fillout NA

#         # O código original estava demorando 4 minutos para ser executado. 
#         # O novo código, utilizando o comando loc e str.split foi executado em 7 segundos.

#         # 08/05/22: Vou usar um dataframe auxiliar para fazer os preenchimentos de células vazias. Tentei substituir os valores no dataframe original e estava tendo problemas.
#         df_aux = df1


#         #competition_distance        

#         df1.loc[df1.competition_distance.isna(),'competition_distance'] = 200000.0

#         #competition_open_since_month

#         df1['competition_open_since_month'] = np.where(df_aux['competition_open_since_month'].isna() == True, df_aux.date.dt.month, df_aux['competition_open_since_month'])

#         #competition_open_since_year 

#         df1['competition_open_since_year'] = np.where(df_aux['competition_open_since_year'].isna() == True, df_aux.date.dt.year, df_aux['competition_open_since_year'])

#         #promo2_since_week          

#         df1['promo2_since_week'] = np.where(df_aux['promo2_since_week'].isna() == True, df_aux.date.dt.week, df_aux['promo2_since_week'])

#         #promo2_since_year    

#         df1['promo2_since_year'] = np.where(df_aux['promo2_since_year'].isna() == True, df_aux.date.dt.year, df_aux['promo2_since_year'])

#         #promo_interval              
#         month_map = {1: 'Jan',  2: 'Feb',  3: 'Mar',  4: 'Apr',  5: 'May',  6: 'Jun',  7: 'Jul',  8: 'Aug',  9: 'Sept',  10: 'Oct', 11: 'Nov', 12: 'Dec'}

#         df1['promo_interval'].fillna(0, inplace=True )

#         df1['month_map'] = df1['date'].dt.month.map( month_map )

#         df1[['m1','m2','m3','m4']] = df1['promo_interval'].str.split( ',' , expand = True)
#         df1['is_promo3'] = 0
#         df1.loc[(df1.month_map == df1.m1) | (df1.month_map == df1.m2 ) | (df1.month_map == df1.m3 ) | (df1.month_map == df1.m4), 'is_promo3' ] = 1

#         df1.drop(columns=['m1','m2','m3','m4'],inplace=True)

#         ## 1.6. Change Data Types

#         # competiton
#         df1['competition_open_since_month'] = df1['competition_open_since_month'].astype( int )
#         df1['competition_open_since_year'] = df1['competition_open_since_year'].astype( int )

#         # promo2
#         df1['promo2_since_week'] = df1['promo2_since_week'].astype( int )
#         df1['promo2_since_year'] = df1['promo2_since_year'].astype( int )
        
#         return df1
    
    
#     def feature_engineering(self,df2):
#         # year
#         df2['year'] = df2['date'].dt.year

#         # month
#         df2['month'] = df2['date'].dt.month

#         # day
#         df2['day'] = df2['date'].dt.day

#         # week of year
#         df2['week_of_year'] = df2['date'].dt.weekofyear

#         # year week
#         df2['year_week'] = df2['date'].dt.strftime( '%Y-%W' )

#         # Tempo de execução: 3.64s
#         # competition since
#         # Criando a data completa de início da competição
#         aux_str = df2['competition_open_since_year'].astype(str) + '-' + df2['competition_open_since_month'].astype(str) + '-1'
#         df2['competition_since'] = pd.to_datetime(aux_str)

#         # Tempo de execução: 367ms
#         aux_delta_days = ( df2['date'] - df2['competition_since'] )
#         df2['competition_time_month'] = (aux_delta_days/30).dt.days

#         # Tempo de execução 3.6s
#         aux_str = df2['promo2_since_year'].astype( str ) + '-' + df2['promo2_since_week'].astype( str ) + '-1'
#         df2['promo_since'] = pd.to_datetime(aux_str,format="%Y-%W-%w") - timedelta(7)

#         # Tempo de execução: 388ms
#         aux_delta_days = ( df2['date'] - df2['promo_since'] )
#         df2['promo_time_week'] = (aux_delta_days/7).dt.days

#         # Tempo de execução: 443ms
#         df2['assortment'] = 'basic'
#         df2.loc[df1.assortment == 'b', 'assortment'] = 'extra'
#         df2.loc[df1.assortment == 'c', 'assortment'] = 'extended'

#         # Tempo de Execução 393ms
#         df2['state_holiday'] = 'regular_day'
#         df2.loc[df1.state_holiday == 'a', 'state_holiday'] = 'public_holiday'
#         df2.loc[df1.state_holiday == 'b', 'state_holiday'] = 'easter_holiday'
#         df2.loc[df1.state_holiday == 'c', 'state_holiday'] = 'christmas'


#         # 3.0. PASSO 03 - FILTRAGEM DE VARIÁVEIS

#         ## 3.1. Filtragem das Linhas


#         df2 = df2[df2['open'] != 0]

#         ## 3.2. Selecao das Colunas

#         cols_drop = ['open', 'promo_interval', 'month_map']
#         df2 = df2.drop( cols_drop, axis=1 )

#         return df2
    
#         # 5.0 Passo 05 - DATA PREPARATION

#     def data_preparation(self,df5):


#         ## 5.2 Rescaling
#         # competition distance
#         df5['competition_distance'] = self.competition_distance_scaler.transform( df5[['competition_distance']].values )


#         # competition time month
#         df5['competition_time_month'] = self.competition_time_month.transform( df5[['competition_time_month']].values )


#         # promo time week
#         df5['promo_time_week'] = self.promo_time_week.transform( df5[['promo_time_week']].values )


#         # year
#         df5['year'] = self.year.transform( df5[['year']].values )

#         ## 5.3 Transformação

#         ### 5.3.1 Encoding

#         # 'state_holiday' - One Hot Encoding
#         df5 = pd.get_dummies(df5,prefix=['state_holiday'],columns=['state_holiday'] )

#         # 'store_type' - Label Encoding
#         le = LabelEncoder()
#         df5['store_type'] = self.store_type_scaler.transform(df5.store_type)

#         # 'assortment' - Ordinal Encoding
#         assortment_dict = {'basic':1,'extra':2,'extended':3}
#         df5['assortment'] = df5.assortment.map(assortment_dict)

#         # state_holiday - One Hot Encoding
#         df5 = pd.get_dummies( df5, prefix=['state_holiday'], columns=['state_holiday'] )


#         ### 5.3.3 Nature Transformation
#         #Transformação de natureza cíclica. Mudei algumas coisas feitas pelo Meigarom.
#         #Xsin = sin(2pi*x/max(x))
#         #Xcos = cos(2pi*x/max(x))

#         df5['month_sin'] = np.sin(df5.month*(2.*np.pi/12))
#         df5['month_cos'] = np.cos(df5.month*(2.*np.pi/12))

#         df5['day_of_week_sin'] = np.sin(df5.day_of_week*(2.*np.pi/7))
#         df5['day_of_week_cos'] = np.cos(df5.day_of_week*(2.*np.pi/7))

#         df5['day_sin'] = np.sin(df5.day*(2.*np.pi/30))
#         df5['day_cos'] = np.cos(df5.day*(2.*np.pi/30))

#         df5['week_of_year_sin'] = np.sin(df5.week_of_year*(2.*np.pi/52))
#         df5['week_of_year_cos'] = np.cos(df5.week_of_year*(2.*np.pi/52))
        
#         cols_selected = ['store', 'promo', 'store_type', 'assortment', 'competition_distance', 'competition_open_since_month', 'competition_open_since_year', 'promo2', 'promo2_since_week', 'promo2_since_year', 'competition_time_month', 'promo_time_week', 'day_of_week_sin', 'day_of_week_cos', 'month_sin', 'month_cos', 'day_sin', 'day_cos', 'week_of_year_sin', 'week_of_year_cos' ]            
#         return df5
    
#     def get_prediction(self,model,original_data,test_data):
#          # prediction
        
#         #join pred into the original data
#         original_data['prediction'] = np.expm1(pred)
        
#         return original_data.to_json(orient='recors',date_format='iso')
    
# ========================================================================================================

# Versão 03 - Atualizado: 03/07/22 - 12h26
# Comentei o self.home_path por causa do erro abaixo no log do heroku:
# 2022-07-03T15:23:11.189330+00:00 app[web.1]:     self.competition_distance_scaler    = pickle.load(  self.home_path + open( 'parameter/competition_distance_scaler.pkl', 'rb') )
# 2022-07-03T15:23:11.189330+00:00 app[web.1]: TypeError: can only concatenate str (not "_io.BufferedReader") to str
# 2022-07-03T15:23:11.189646+00:00 app[web.1]: 10.1.60.250 - - [03/Jul/2022 15:23:11] "POST /rossmann/predict HTTP/1.1" 500 -

# UPDATE: Novo problema, agora no timedelta

# 2022-07-03T15:37:06.611275+00:00 app[web.1]:   File "/app/rossmann/Rossmann.py", line 127, in feature_engineering
# 2022-07-03T15:37:06.611275+00:00 app[web.1]:     df2['promo_since'] = pd.to_datetime(aux_str,format="%Y-%W-%w") - timedelta(7)
# 2022-07-03T15:37:06.611275+00:00 app[web.1]: NameError: name 'timedelta' is not defined

# 03/07/22 - 12h52: Importei datetime e alterei timedelta para 

# 03/07/22 - 13h01: Acrescentei df1 = df2.copy() na linha 111. Tirei biblioteca do Boruta e jupyter

# 13h08:
# 2022-07-03T16:06:32.975364+00:00 app[web.1]:   File "/app/rossmann/Rossmann.py", line 183, in data_preparation
# 2022-07-03T16:06:32.975365+00:00 app[web.1]:     df5['competition_time_month'] = self.competition_time_month.transform( df5[['competition_time_month']].values )
# 2022-07-03T16:06:32.975365+00:00 app[web.1]: AttributeError: 'Rossmann' object has no attribute 'competition_time_month'

# 13h04
# 2022-07-03T16:12:22.450573+00:00 app[web.1]:   File "/app/rossmann/Rossmann.py", line 206, in data_preparation
# 2022-07-03T16:12:22.450573+00:00 app[web.1]:     le = LabelEncoder()
# 2022-07-03T16:12:22.450574+00:00 app[web.1]: NameError: name 'LabelEncoder' is not defined

# 13h20
# 2022-07-03T16:17:47.027835+00:00 app[web.1]:   File "handler.py", line 34, in rossmann_predict
# 2022-07-03T16:17:47.027835+00:00 app[web.1]:     df3 = pipeline.data_preparation( df2 )
# 2022-07-03T16:17:47.027835+00:00 app[web.1]:   File "/app/rossmann/Rossmann.py", line 218, in data_preparation
# 2022-07-03T16:17:47.027836+00:00 app[web.1]:     df5 = pd.get_dummies( df5, prefix=['state_holiday'], columns=['state_holiday'] )

import pickle 
import inflection
import pandas as pd
import numpy as np
import math
import datetime

class Rossmann(object):
    def __init__(self):
        # Versão Heroku
        #self.home_path=''
        # self.competition_distance_scaler    = pickle.load( open( 'parameter/competition_distance_scaler.pkl', 'rb') )
        # self.competition_time_month_scaler  = pickle.load( open( 'parameter/competition_time_month_scaler.pkl', 'rb') )
        # self.promo_time_week_scaler         = pickle.load( open( 'parameter/promo_time_week_scaler.pkl', 'rb') )
        # self.year_scaler                    = pickle.load( open( 'parameter/year_scaler.pkl', 'rb') )
        # self.store_type_scaler              = pickle.load( open( 'parameter/store_type_scaler.pkl', 'rb') )

        # Versão Local
        self.home_path='/Users/igor/repos/ds-em-producao/'
        self.competition_distance_scaler    = pickle.load( open( self.home_path + 'parameter/competition_distance_scaler.pkl', 'rb') )
        self.competition_time_month_scaler  = pickle.load( open( self.home_path + 'parameter/competition_time_month_scaler.pkl', 'rb') )
        self.promo_time_week_scaler         = pickle.load( open( self.home_path + 'parameter/promo_time_week_scaler.pkl', 'rb') )
        self.year_scaler                    = pickle.load( open( self.home_path + 'parameter/year_scaler.pkl', 'rb') )
        self.store_type_scaler              = pickle.load( open( self.home_path + 'parameter/store_type_scaler.pkl', 'rb') )
    

    def data_cleaning(self,df1):
        
        ## 1.1. Rename Columns

        cols_old = ['Store', 'DayOfWeek', 'Date', 'Open', 'Promo', 'StateHoliday', 'SchoolHoliday', 
                    'StoreType', 'Assortment', 'CompetitionDistance', 'CompetitionOpenSinceMonth',
                    'CompetitionOpenSinceYear', 'Promo2', 'Promo2SinceWeek', 'Promo2SinceYear', 'PromoInterval']

        snakecase = lambda x: inflection.underscore( x )

        cols_new = list( map( snakecase, cols_old ) )

        # rename
        df1.columns = cols_new


        ## 1.3. Data Types

        df1['date'] = pd.to_datetime( df1['date'] )
        

        ## 1.5. Fillout NA

        # O código original estava demorando 4 minutos para ser executado. 
        # O novo código, utilizando o comando loc e str.split foi executado em 7 segundos.

        # 08/05/22: Vou usar um dataframe auxiliar para fazer os preenchimentos de células vazias. Tentei substituir os valores no dataframe original e estava tendo problemas.
        df_aux = df1


        #competition_distance        

        df1.loc[df1.competition_distance.isna(),'competition_distance'] = 200000.0

        #competition_open_since_month

        df1['competition_open_since_month'] = np.where(df_aux['competition_open_since_month'].isna() == True, df_aux.date.dt.month, df_aux['competition_open_since_month'])

        #competition_open_since_year 

        df1['competition_open_since_year'] = np.where(df_aux['competition_open_since_year'].isna() == True, df_aux.date.dt.year, df_aux['competition_open_since_year'])

        #promo2_since_week          

        df1['promo2_since_week'] = np.where(df_aux['promo2_since_week'].isna() == True, df_aux.date.dt.week, df_aux['promo2_since_week'])

        #promo2_since_year    

        df1['promo2_since_year'] = np.where(df_aux['promo2_since_year'].isna() == True, df_aux.date.dt.year, df_aux['promo2_since_year'])

        #promo_interval              
        month_map = {1: 'Jan',  2: 'Feb',  3: 'Mar',  4: 'Apr',  5: 'May',  6: 'Jun',  7: 'Jul',  8: 'Aug',  9: 'Sept',  10: 'Oct', 11: 'Nov', 12: 'Dec'}

        df1['promo_interval'].fillna(0, inplace=True )

        df1['month_map'] = df1['date'].dt.month.map( month_map )

        df1[['m1','m2','m3','m4']] = df1['promo_interval'].str.split( ',' , expand = True)
        df1['is_promo3'] = 0
        df1.loc[(df1.month_map == df1.m1) | (df1.month_map == df1.m2 ) | (df1.month_map == df1.m3 ) | (df1.month_map == df1.m4), 'is_promo3' ] = 1

        df1.drop(columns=['m1','m2','m3','m4'],inplace=True)

        ## 1.6. Change Data Types

        # competiton
        df1['competition_open_since_month'] = df1['competition_open_since_month'].astype( int )
        df1['competition_open_since_year'] = df1['competition_open_since_year'].astype( int )

        # promo2
        df1['promo2_since_week'] = df1['promo2_since_week'].astype( int )
        df1['promo2_since_year'] = df1['promo2_since_year'].astype( int )
        
        return df1
    
    
    def feature_engineering(self,df2):
        df1 = df2.copy()

        # year
        df2['year'] = df2['date'].dt.year

        # month
        df2['month'] = df2['date'].dt.month

        # day
        df2['day'] = df2['date'].dt.day

        # week of year
        df2['week_of_year'] = df2['date'].dt.weekofyear

        # year week
        df2['year_week'] = df2['date'].dt.strftime( '%Y-%W' )

        # Tempo de execução: 3.64s
        # competition since
        # Criando a data completa de início da competição
        aux_str = df2['competition_open_since_year'].astype(str) + '-' + df2['competition_open_since_month'].astype(str) + '-1'
        df2['competition_since'] = pd.to_datetime(aux_str)

        # Tempo de execução: 367ms
        aux_delta_days = ( df2['date'] - df2['competition_since'] )
        df2['competition_time_month'] = (aux_delta_days/30).dt.days

        # Tempo de execução 3.6s
        aux_str = df2['promo2_since_year'].astype( str ) + '-' + df2['promo2_since_week'].astype( str ) + '-1'
        #df2['promo_since'] = pd.to_datetime(aux_str,format="%Y-%W-%w") - timedelta(7)
        df2['promo_since'] = pd.to_datetime(aux_str,format="%Y-%W-%w") - datetime.timedelta( days=7) 
        # Tempo de execução: 388ms
        aux_delta_days = ( df2['date'] - df2['promo_since'] )
        df2['promo_time_week'] = (aux_delta_days/7).dt.days

        # Tempo de execução: 443ms
        df2['assortment'] = 'basic'
        df2.loc[df1.assortment == 'b', 'assortment'] = 'extra'
        df2.loc[df1.assortment == 'c', 'assortment'] = 'extended'

        # Tempo de Execução 393ms
        df2['state_holiday'] = 'regular_day'
        df2.loc[df1.state_holiday == 'a', 'state_holiday'] = 'public_holiday'
        df2.loc[df1.state_holiday == 'b', 'state_holiday'] = 'easter_holiday'
        df2.loc[df1.state_holiday == 'c', 'state_holiday'] = 'christmas'


        # 3.0. PASSO 03 - FILTRAGEM DE VARIÁVEIS

        ## 3.1. Filtragem das Linhas


        df2 = df2[df2['open'] != 0]

        ## 3.2. Selecao das Colunas

        cols_drop = ['open', 'promo_interval', 'month_map']
        df2 = df2.drop( cols_drop, axis=1 )

        return df2
    
        # 5.0 Passo 05 - DATA PREPARATION

    def data_preparation(self,df5):


        ## 5.2 Rescaling
        # competition distance
        df5['competition_distance'] = self.competition_distance_scaler.transform( df5[['competition_distance']].values )


        # competition time month
        df5['competition_time_month'] = self.competition_time_month_scaler.transform( df5[['competition_time_month']].values )


        # promo time week
        df5['promo_time_week'] = self.promo_time_week_scaler.transform( df5[['promo_time_week']].values )


        # year
        df5['year'] = self.year_scaler.transform( df5[['year']].values )

        ## 5.3 Transformação

        ### 5.3.1 Encoding

        # 'state_holiday' - One Hot Encoding
        df5 = pd.get_dummies(df5,prefix=['state_holiday'],columns=['state_holiday'] )

        # 'store_type' - Label Encoding
        df5['store_type'] = self.store_type_scaler.transform(df5.store_type)

        # 'assortment' - Ordinal Encoding
        assortment_dict = {'basic':1,'extra':2,'extended':3}
        df5['assortment'] = df5.assortment.map(assortment_dict)


        ### 5.3.3 Nature Transformation
        #Transformação de natureza cíclica. Mudei algumas coisas feitas pelo Meigarom.
        #Xsin = sin(2pi*x/max(x))
        #Xcos = cos(2pi*x/max(x))

        df5['month_sin'] = np.sin(df5.month*(2.*np.pi/12))
        df5['month_cos'] = np.cos(df5.month*(2.*np.pi/12))

        df5['day_of_week_sin'] = np.sin(df5.day_of_week*(2.*np.pi/7))
        df5['day_of_week_cos'] = np.cos(df5.day_of_week*(2.*np.pi/7))

        df5['day_sin'] = np.sin(df5.day*(2.*np.pi/30))
        df5['day_cos'] = np.cos(df5.day*(2.*np.pi/30))

        df5['week_of_year_sin'] = np.sin(df5.week_of_year*(2.*np.pi/52))
        df5['week_of_year_cos'] = np.cos(df5.week_of_year*(2.*np.pi/52))
        
        cols_selected = ['store', 'promo', 'store_type', 'assortment', 'competition_distance', 'competition_open_since_month', 'competition_open_since_year', 'promo2', 'promo2_since_week', 'promo2_since_year', 'competition_time_month', 'promo_time_week', 'day_of_week_sin', 'day_of_week_cos', 'month_sin', 'month_cos', 'day_sin', 'day_cos', 'week_of_year_sin', 'week_of_year_cos' ]            
        return df5[cols_selected]
    
    def get_prediction(self,model,original_data,test_data):
         # prediction
        pred = model.predict( test_data )
        
        #join pred into the original data
        original_data['prediction'] = np.expm1(pred)
        
        return original_data.to_json(orient='records',date_format='iso')