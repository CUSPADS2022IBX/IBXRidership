peak = ['exits_weekday_morning','entries_weekday_morning','exits_weekday_evening','entries_weekday_evening']

cols_offpeak = ['cs_noibx_offpeak','lotarea', 'bldgarea',
                'comarea', 'resarea', 'officearea', 'retailarea', 'garagearea',
                'strgearea', 'factryarea', 'otherarea', 'numbldgs', 'numfloors',
                'unitsres', 'unitstotal', 'assesstot', 'Total_pop_Num',
                'housed_pop_Num', 'group_house_pop_num', 'industrial_pop_num',
                'under18_num', 'hispanic_num', 'white_num', 'black_num', 'asian_num',
                'other_num', 'multirace_num', 'total_houses', 'occupied_homes_num',
                'vacant_homes_num', 'Bus_Stops']

cols_peak = ['cs_noibx_peak','lotarea', 'bldgarea',
             'comarea', 'resarea', 'officearea', 'retailarea', 'garagearea',
             'strgearea', 'factryarea', 'otherarea', 'numbldgs', 'numfloors',
             'unitsres', 'unitstotal', 'assesstot', 'Total_pop_Num',
             'housed_pop_Num', 'group_house_pop_num', 'industrial_pop_num',
             'under18_num', 'hispanic_num', 'white_num', 'black_num', 'asian_num',
             'other_num', 'multirace_num', 'total_houses', 'occupied_homes_num',
             'vacant_homes_num', 'Bus_Stops']

#feed in entries and exits as strings (column names)
def prediction_tables(entries,exits,df_train,df_ibx):
  #check if peak or off peak 
  if entries in peak:
    #entries
    model = LassoCV(cv=5, random_state=0).fit(df_train[cols_peak], df_train[entries])
    #print('{}:'.format(entries), Lasso_model.score(df_train[cols_peak], df_train[entries]))
    entries_preds = model.predict(df_ibx[cols_peak])

    #exits
    model = LassoCV(cv=5, random_state=0).fit(df_train[cols_peak], df_train[exits])
    #print('{}:'.format(exits), Lasso_model.score(df_train[cols_peak], df_train[exits]))
    exits_preds = model.predict(df_ibx[cols_peak])
  else:
    model = LassoCV(cv=5, random_state=0).fit(df_train[cols_offpeak], df_train[entries])
    #print('{}:'.format(entries), Lasso_model.score(df_train[cols_offpeak], df_train[entries]))
    entries_preds = model.predict(df_ibx[cols_offpeak])

    #exits
    model = LassoCV(cv=5, random_state=0).fit(df_train[cols_offpeak], df_train[exits])
    #print('{}:'.format(exits), Lasso_model.score(df_train[cols_offpeak], df_train[exits]))
    exits_preds = model.predict(df_ibx[cols_offpeak])
  
  #now merging so voro id, complex id, entries, exits in single df 
  vals = pd.concat([df_ibx['Complex ID'], df_ibx['VoroID'],pd.Series(entries_preds),pd.Series(exits_preds)], axis=1)
  return vals

