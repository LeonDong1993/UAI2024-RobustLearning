import numpy as np 
import pandas as pd 
import glob 

def parse_row(row_str):
    tokens = row_str.strip().split(':')[1:]
    values = [t.split(' ')[0] for t in tokens]
    values = list(map(float, values))
    return values[-2:]

def summarize(result_dir, metric_names, method_names, with_std = True):
    table = []
    for file in glob.glob(f'{result_dir}/*.log'):
        fname = file.split('/')[-1]
        data_name = fname.split('_')[0]
        with open(file, 'r') as f:
            content = f.readlines()[2:]
        
        assert( len(content) == len(method_names) * len(metric_names) ), "Input size not match"

        
        i = 0
        for method in method_names:
            row = [data_name, method]
            for metric in metric_names:
                # parse the mean and std 
                mean, std = parse_row(content[i])
                if data_name == 'mnist' and 'NB' in metric:
                    mean, std = 0,0

                item = f'{mean:.2f} ± {std:.1f}' if with_std else mean
                row.append(item)
                i += 1
            table.append(row)
        
    df = pd.DataFrame(table, columns=['Dataset', 'Metric'] + metric_names)
    df.sort_values(by = ['Dataset', 'Metric'], inplace=True)
    return df


def compute_average_and_nwin(df):
    df_mean = df.groupby('Metric').mean()
    df_drsl = df[df['Metric'] == 'DRSL']
    df_mle = df[df['Metric'] == 'MLE']
    mle_arr = df_mle.values[:,2:].astype(float)
    drsl_arr= df_drsl.values[:,2:].astype(float)
    mle_win = (mle_arr >= drsl_arr + 1e-2).sum(axis = 0)
    drsl_win = (drsl_arr >= mle_arr + 1e-2).sum(axis = 0)

    # correction for mnist dataset 
    df_mean['Worst NB'] = df_mean['Worst NB'] *9.0/8
    df_mean['Avg. NB'] = df_mean['Avg. NB'] *9.0/8
    
    mle_win[-2:] -= 1
    drsl_win[-2:] -= 1

    print(df_mean.to_csv(index=False))
    print(','.join(map(str, drsl_win)))
    print(','.join(map(str, mle_win)))
    

if __name__ == '__main__':
    metric_names = ['Ori. Test', 'Gau. Test', 'Jit. Test', 'Worst NB', 'Avg. NB']
    df_std = summarize('./results/nngbn/', metric_names, ['DRSL', 'MLE'])
    print(df_std.to_csv(index = False))
    df = summarize('./results/nngbn/', metric_names, ['DRSL', 'MLE'], with_std=False)
    compute_average_and_nwin(df)

    df_std = summarize('./results/mixmg-auto3to9/', metric_names, ['MLE', 'DRSL'])
    print(df_std.to_csv(index = False))
    df = summarize('./results/mixmg-auto3to9/', metric_names, ['MLE', 'DRSL'], with_std=False)
    compute_average_and_nwin(df)
    
    
