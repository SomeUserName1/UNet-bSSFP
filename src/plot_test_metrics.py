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
    for root, dirs, _ in os.walk('/ptmp/fklopfer/logs/finetune'):
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
               ['dxx_norm', 'dxy_norm', 'dxz_norm', 'dyy_norm', 'dyz_norm', 'dzz_norm'],
                ['fa', 'md', 'ad', 'rd', 'inclination', 'azimuth']
               ]
    col_names = ['denorm_tensor', 'norm_tensor', 'scalars']
    flat_cols = list(chain.from_iterable(columns))
    for col in flat_cols:
       #  ax = df.hist(column=col, by=by, sharex=True, sharey=True, xrot=90, figsize=(27, 36),
       #          range=(0, 10), bins=100, legend=True, density=True, stacked=True)
       #  axs = ax.flatten()
       #  for a in axs:
       #      a.set_xscale('log')
       #  fig = ax[0][0].figure
       #  fig.tight_layout()
       #  fig.savefig(f'err_{name}_{datetime.datetime.now()}.pdf')
        ax = sns.violinplot(df, log_scale=True, x='roi', y=col, hue='modality', split=True)
        fig = ax.figure.savefig(f'violins_{col}.pdf')
        plt.clf()

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
    diag_norm = [x + '_norm' for x in diag]
    non_diag_norm = [x + '_norm' for x in non_diag]

    modalities = ['DTI', 'pc-bSSFP', 'bSSFP', 'T1w']
    cols = ['CSF', 'WM', 'GM']
    df['Modality'].replace({'bssfp': 'bSSFP', 'dwi-tensor': 'DTI', 'pc-bssfp': 'pc-bSSFP', 't1w': 'T1w'}, inplace=True)

    for d, nd, pref in zip([diag, diag_norm], [non_diag, non_diag_norm], ['', 'Normalized ']):
        df_diag = df.loc[df['Scalar'].isin(d)]
        df_non_diag = df.loc[df['Scalar'].isin(nd)]
        diag_group = df_diag.groupby(by=['ROI', 'Modality']).mean(numeric_only=True)
        non_diag_group = df_non_diag.groupby(by=['ROI', 'Modality']).mean(numeric_only=True)
        diag_mean = diag_group['Mean'].unstack().T.reindex(modalities).reset_index()
        non_diag_mean = non_diag_group['Mean'].unstack().T.reindex(modalities).reset_index()
        diag_std = diag_group['Std'].unstack().T.reindex(modalities).reset_index()
        non_diag_std = non_diag_group['Std'].unstack().T.reindex(modalities).reset_index()

        fig, (ax1, ax2) = plt.subplots(1, 2, sharey=True, figsize=(8, 4))
        fig.suptitle('Mean relative Error of elements in the' + pref + 'Diffusion Tensor')
        diag_mean.plot.bar(ax=ax1, title='Diagonal',  x='Modality', y=['CSF', 'GM', 'WM'], rot=0, yerr=diag_std.set_index('Modality'), capsize=2)
        non_diag_mean.plot.bar(ax=ax2, title='Off-diagonal', x='Modality', y=['CSF', 'GM', 'WM'], rot=0, yerr=non_diag_std.set_index('Modality'), capsize=2)
        plt.tight_layout()
        fig.savefig(pref[:-1] + '_tensor_errs.pdf')


def plot_stacked_bar_scalars():
    csv_name = '/home/fklopfer/UNet-bSSFP/src/sample_stats.csv'
    df = pd.read_csv(csv_name)

    pretty_names = ['Fractional Anisotropy', 'Mean Diffusivity', 'Axial Diffusivity', 'Radial Diffusivity', 'Inclination', 'Azimuth']
    scalars = ['fa', 'md', 'ad', 'rd', 'inclination', 'azimuth']
    modalities = ['DTI', 'pc-bSSFP', 'bSSFP', 'T1w']

    cols = ['CSF', 'WM', 'GM']

    df['Modality'].replace({'bssfp': 'bSSFP', 'dwi-tensor': 'DTI', 'pc-bssfp': 'pc-bSSFP', 't1w': 'T1w'}, inplace=True)
    for s, ps in zip(scalars, pretty_names):
        df_scalar = df.loc[df['Scalar'] == s].set_index(['ROI', 'Modality'])
        scalar_mean = df_scalar['Mean'].unstack().T.reindex(modalities).reset_index()
        scalar_std = df_scalar['Std'].unstack().T.reindex(modalities).reset_index()

        fig, ax = plt.subplots()
        fig.suptitle('Mean relative Error of ' + ps)
        scalar_mean.plot.bar(ax=ax, x='Modality', y=['CSF', 'GM', 'WM'], rot=0, yerr=scalar_std.set_index('Modality'), capsize=2)
        plt.tight_layout()
        fig.savefig(s + '_errs.pdf')

if __name__ == '__main__':
    plot_nn_metrics()
    plot_stacked_bar_tensors()
    plot_stacked_bar_scalars()
