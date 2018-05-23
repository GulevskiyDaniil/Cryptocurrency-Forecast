import pandas as pd
import datetime
import argparse


# -i input_file, -o output_file  -b begin_ts -e end_ts

def to_timestamp(df):
    df['T'] = df['T'].map(lambda string: datetime.datetime.strptime(string,
      '%Y-%m-%dT%H:%M:%S'))

#def main():
parser = argparse.ArgumentParser()
parser.add_argument('-i', type=str, nargs='?')
parser.add_argument('-o', type=str, nargs='?')
parser.add_argument('-b', type=str, nargs='?')
parser.add_argument('-e', type=str, nargs='?')
parser.add_argument('-c', type=str, nargs='+')
parser.add_argument('-s', action='store_true')



args = parser.parse_args()

input_fname = args.i
output_fname = args.o
begin = datetime.datetime.strptime(args.b, '%Y.%m.%d.%H.%M.%S')
end = datetime.datetime.strptime(args.e, '%Y.%m.%d.%H.%M.%S')
columns = args.c
standardize = args.s

tables = []
index = []
time_index_saved = False

with open(input_fname) as infile:
    for i, line in enumerate(infile):
        data_fname = line.strip()

        data = pd.read_csv(data_fname, delimiter=';', decimal=',')
        to_timestamp(data)
        if data['T'].iloc[0] > begin or data['T'].iloc[-1] < end:
            print("File {} doenst contain all data for the period of interest".
                  format(data_fname))
            continue
        for colname in columns:
            index.append('{}:{}'.format(data_fname.split('/')[-1],
                         colname))
        data = data[(data['T'] >= begin) & (data['T'] <= end)]
        data.set_index('T', inplace=True)
        data = data[columns]
        data.columns = ['{}_{}'.format(data_fname.split('/')[-1].split('_')[0], colname)
                for colname in data.columns]
        
        if i == 0:
            big_table = data
        else:
            big_table = pd.merge(big_table, data.copy(), how='outer',
                                 left_index=True, right_index=True)
        
        
for col in big_table.columns:
    if big_table[col].isnull().sum() > 0:
        print("Some time moments are missing from {}".format(col))


if standardize:
    print("Standardizing..")
    big_table=(big_table-big_table.mean(0))/big_table.std(0)


big_table.T.to_csv(output_fname, na_rep='n', header=False, index=False)
time_index_file_name = output_fname.split('.')[0] + '_time.csv'
big_table.index.to_series().to_csv(time_index_file_name)

index_file_name = output_fname.split('.')[0] + '_series.csv'
pd.Series(index).to_csv(index_file_name, index=False)



