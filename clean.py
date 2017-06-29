import pandas as pd

main_df = pd.read_csv('cdc.csv')
main_df = main_df.set_index(['Geography', 'Year'])
print main_df.dtypes #check if year is string or int

search_terms = ['TB', 'cough', 'tuberculosis', 'sex']

#this concatenates the CDC data file with the Google Trends files. The Google Trends files are saved in the format '[search_term][year].csv'.
for phrase in search_terms:
    terms_df_list = []
    for i in range(4, 16):
        readfile = phrase + str(i).zfill(2) + '.csv'
        lines = open(readfile).readlines()
        new_readfile = 'processed_' + readfile
        open(new_readfile, 'w').writelines(lines[3:])
        df = pd.read_csv(new_readfile, names=['Geography', phrase])
        year = int('20' + str(i).zfill(2))
        df['Year'] = year
        df = df.set_index(['Geography', 'Year'])
        terms_df_list.append(df)
    terms_df = pd.concat(terms_df_list)
    main_df = main_df.join(terms_df)

main_df = main_df.dropna() #drop non-50 states rows---these have NaN because they are not in Google Trends data

#process 'Rates' column as follows. First convert to string, remove commas, then convert to int.
def conv_num(cell):
    cell = cell.replace(",", "")
    return int(cell)

main_df['Cases'] = main_df['Cases'].apply(conv_num)

#calculate columns for possible other kinds of rates to predict.
main_df['year_tot_cases'] = main_df.groupby(level='Year')['Cases'].transform('sum')
main_df['prop_of_year_tot'] = main_df['Cases']/main_df['year_tot_cases']

#save to file for processing by training script.
main_df.to_csv('all_data.csv')
