import pandas as pd
import numpy as np
from catboost import CatBoostRegressor
from sklearn.model_selection import GridSearchCV, KFold

np.random.seed(123)

mdir = '/gpfs/data1/cmongp1/Guanyuan/County_yield/Acres/annual'
df = pd.read_csv(f'{mdir}/united_states_of_america_Corn_acres_final.csv')
df = df[df['Year'] > 2002].dropna()

df['global_trend'] = np.polyval(np.polyfit(df['Year'], df['yield'], 1), df['Year'])
df['yield_anomaly'] = df['yield'] - df['global_trend']

cols_to_remove = df.columns[df.columns.str.contains('biweek(1|2|3|4|5|6|7|8|17|18|19|20|21|22|23|24|25|26)$')]
df = df.drop(columns=cols_to_remove)

df['states'] = df['FIPS'] // 1000
MW_states = [17, 18, 19, 20, 26, 27, 29, 31, 38, 39, 46, 55]
# df = df[df['states'].isin(MW_states) & (df['Year'] < 2017)]
df = df.drop(columns=['states'])

dependent_variable_name = 'yield_anomaly'
predictors = [col for col in df.columns if col.startswith(('esi', 'pr', 'gcvi'))]

ys = sorted(df['Year'].unique())

results = []

for year in ys:
    print(f"Processing year: {year}")
    
    df_train = df[df['Year'] != year]
    df_test = df[df['Year'] == year]
    
    X_train = df_train[predictors]
    y_train = df_train[dependent_variable_name]
    X_test = df_test[predictors]
    
    model = CatBoostRegressor(silent=True)
    
    param_grid = {
        'depth': [6, 8, 10],
        'learning_rate': [0.01, 0.1, 0.2],
        'iterations': [100, 200, 500,1000,2000]
    }
    
    cv = KFold(n_splits=5, shuffle=True, random_state=123)
    grid_search = GridSearchCV(estimator=model, param_grid=param_grid, cv=cv, n_jobs=10, verbose=1)
    grid_search.fit(X_train, y_train)
    
    predictions = grid_search.best_estimator_.predict(X_test)
    
    df_test['predict'] = predictions
    df_test = df_test[['predict', 'Year', 'yield', 'yield_anomaly', 'global_trend', 'FIPS']]
    
    results.append(df_test)

out_df = pd.concat(results, ignore_index=True)
out_df.to_csv(f'{mdir}/Yield_prediction_rs_RF_detrend_ESI_PR_GCVI_tuning_catboost.csv', index=False)
