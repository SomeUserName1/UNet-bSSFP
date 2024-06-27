import matplotlib.pyplot as plt
import os
import pandas as pd
import seaborn as sns
from tabulate import tabulate
import datetime
from itertools import chain


def plot_nn_metrics():
    csv_fname = 'test_metrics.csv'
    keys = []
    metrics = pd.DataFrame()
    for root, dirs, _ in os.walk('/ptmp/fklopfer/logs/direct/'):
        for dir_name in dirs:
            for _, _, fnames in os.walk(os.path.join(root, dir_name)):
                if csv_fname in fnames:
                    row = pd.read_csv(os.path.join(root, dir_name, csv_fname))
                    row['modality'] = dir_name.split('/')[-1]
                    metrics = pd.concat([metrics, row])

    metrics.drop(['epoch', 'step', 'test_metric_MSEMetric'], axis=1, inplace=True)
    metrics.set_index('modality', inplace=True)
    reorder = ['dwi', 'pc-bssfp', 'one-bssfp', 't1w']
    metrics = metrics.reindex(reorder)
    metrics.to_csv('test_metrics_all.csv')
    metrics.rename(columns={'test_loss_L1': 'L1', 'test_loss_Perceptual': 'Perceptual Loss', 'test_loss_SSIM': 'SSIM Loss', 'test_metric_PSNRMetric': 'PSNR'}, inplace=True)
    metrics.rename({'one-bssfp': 'bSSFP', 'dwi': 'DTI', 'pc-bssfp': 'pc-bSSFP', 't1w': 'T1w'}, inplace=True)
    print(metrics)

    loss = metrics[['L1', 'Perceptual Loss', 'SSIM Loss']]
    psnr = metrics[['PSNR']]

    ax = loss.plot.bar(title='Test Loss', stacked=True, rot=0)
    ax.figure.savefig('test_loss.pdf')

#    ax = errors.plot.bar(title='Test Metrics: Mean Absolute Error', rot=0)
#    ax.figure.savefig('test_mse_mae.pdf')
#
#    ax = ssim.plot.bar(title='Test Metrics: Structual Similarity Index Measure', ylim=(0.95, 1), rot=0)
#    ax.figure.savefig('test_ssim.pdf')
#
    ax = psnr.plot.bar(title='Test Metrics: Peak Signal to Noise Ratio', ylim=(30, 45), rot=0)
    ax.figure.savefig('test_psnr.pdf')


def plot_rel_errors():
    csv_fname = '/home/fklopfer/relative_errors.csv'
    df = pd.read_csv(csv_fname)
#    print(f'columns {df.columns}')
#    print(tabulate(df, headers='keys', tablefmt='psql'))
    
    by = ['roi', 'modality']
    columns = [['dxx', 'dxy', 'dxz', 'dyy', 'dyz', 'dzz'],
                ['fa', 'md', 'ad', 'rd', 'inclination', 'azimuth']
               ]
    col_names = ['norm_tensor', 'scalars']

    stat_df = pd.DataFrame(columns=['Scalar', 'ROI', 'Modality', 'Median', '1-Percentile', '25-Percentile',
        '75-Percentile', '99-Percentile', 'Mean', 'Std', 'Max', 'Min'])
    # extract median per quantity, modality and roi
    flat_cols = list(chain.from_iterable(columns))
    for col in flat_cols:
        data = {}
        group = df[by + [col]].groupby(by)
        data['Scalar'] = [col for i in range(len(group))]
        med_df =  group.median().reset_index()
        data['ROI'] = med_df['roi']
        data['Modality'] = med_df['modality']
        data['Median'] = med_df[col].values
        data['1-Percentile'] = group.quantile(0.01)[col].values
        data['25-Percentile'] = group.quantile(0.25)[col].values
        data['75-Percentile'] = group.quantile(0.75)[col].values
        data['99-Percentile'] = group.quantile(0.99)[col].values
        data['Mean'] = group.mean()[col].values
        data['Std'] = group.std()[col].values
        data['Max'] = group.max()[col].values
        data['Min'] = group.min()[col].values
        row = pd.DataFrame.from_dict(data)
        stat_df = pd.concat([stat_df, row])

    stat_df.set_index(['Scalar', 'ROI', 'Modality'], inplace=True)
    print(tabulate(stat_df, headers='keys', tablefmt='psql'))
    stat_df.to_csv('sample_stats.csv')
    ax = stat_df.plot()
    fig = ax.figure.savefig('stats.pdf')

def plot_stacked_bar_tensors():
    csv_name = '/home/fklopfer/UNet-bSSFP/src/sample_stats.csv'
    df = pd.read_csv(csv_name)
    
    diag = ['dxx', 'dyy', 'dzz']
    non_diag = ['dxy', 'dxz', 'dyz']

    modalities = ['DTI', 'pc-bSSFP', 'bSSFP', 'T1w']
    cols = ['CSF', 'WM', 'GM']
    df['Modality'].replace({'bssfp': 'bSSFP', 'dwi-tensor': 'DTI', 'pc-bssfp': 'pc-bSSFP', 't1w': 'T1w'}, inplace=True)
    als = ['left', 'center', 'right'] * 4
    for d, nd, pref in zip([diag], [non_diag], ['normalized ']):
        df_diag = df.loc[df['Scalar'].isin(d)]
        df_diag.loc[:, 'Median'] = df_diag.loc[:, 'Median'] * 100
        df_non_diag = df.loc[df['Scalar'].isin(nd)]
        df_non_diag.loc[:, 'Median'] = df_non_diag.loc[:, 'Median'] * 100
        diag_group = df_diag.groupby(by=['ROI', 'Modality']).mean(numeric_only=True)
        non_diag_group = df_non_diag.groupby(by=['ROI', 'Modality']).mean(numeric_only=True)
        diag_mean = diag_group['Median'].unstack().T.reindex(modalities).reset_index()
        non_diag_mean = non_diag_group['Median'].unstack().T.reindex(modalities).reset_index()

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 4))
        fig.suptitle('Mean relative Error of elements in the ' + pref + 'Diffusion Tensor in %')
        diag_mean.plot.bar(ax=ax1, title='Diagonal',  x='Modality', y=['CSF', 'GM', 'WM'], rot=0)
        non_diag_mean.plot.bar(ax=ax2, title='Off-diagonal', x='Modality', y=['CSF', 'GM', 'WM'], rot=0, logy=True)

        plt.tight_layout()
        fig.savefig(pref[:-1] + '_tensor_errs.pdf')


def plot_stacked_bar_scalars():
    csv_name = '/home/fklopfer/UNet-bSSFP/src/sample_stats.csv'
    df = pd.read_csv(csv_name)

    pretty_names = ['Fractional Anisotropy', 'Mean Diffusivity', 'Axial Diffusivity', 'Radial Diffusivity', 'Inclination', 'Azimuth']
    scalars = ['fa', 'md', 'ad', 'rd', 'inclination', 'azimuth']
    modalities = ['DTI', 'pc-bSSFP', 'bSSFP', 'T1w']

    cols = ['CSF', 'WM', 'GM']
    als = ['left', 'center', 'right'] * 4
    df['Modality'].replace({'bssfp': 'bSSFP', 'dwi-tensor': 'DTI', 'pc-bssfp': 'pc-bSSFP', 't1w': 'T1w'}, inplace=True)
    for s, ps in zip(scalars, pretty_names):
        df_scalar = df.loc[df['Scalar'] == s].set_index(['ROI', 'Modality'])
        if s not in ['azimuth', 'inclination']:
            df_scalar.loc[:, 'Median'] = df_scalar.loc[:, 'Median'] * 100

        scalar_mean = df_scalar['Median'].unstack().T.reindex(modalities).reset_index()

        fig, ax = plt.subplots()
        if s not in ['azimuth', 'inclination']:
            fig.suptitle('Mean relative Error of ' + ps + ' in %')
        else:
            fig.suptitle('Absolute Error of ' + ps + ' in Degree')

        scalar_mean.plot.bar(ax=ax, x='Modality', y=['CSF', 'GM', 'WM'], rot=0)
        plt.tight_layout()
        fig.savefig(s + '_errs.pdf')

if __name__ == '__main__':
    plot_nn_metrics()
    plot_rel_errors()
    plot_stacked_bar_tensors()
    plot_stacked_bar_scalars()
