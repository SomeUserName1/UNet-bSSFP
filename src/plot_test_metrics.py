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
    print(metrics)
    metrics.to_csv('test_metrics_all.csv')

    loss = metrics[['test_loss_L1', 'test_loss_Perceptual', 'test_loss_SSIM']]
    errors = metrics[['test_metric_MAEMetric']]
    ssim = metrics[['test_metric_SSIMMetric']]
    psnr = metrics[['test_metric_PSNRMetric']]

    ax = loss.plot.bar(title='Test Loss', stacked=True, rot=0)
    ax.figure.savefig('test_loss.pdf')

    ax = errors.plot.bar(title='Test Metrics: Mean Absolute Error', rot=0)
    ax.figure.savefig('test_mse_mae.pdf')

    ax = ssim.plot.bar(title='Test Metrics: Structual Similarity Index Measure', ylim=(0.95, 1), rot=0)
    ax.figure.savefig('test_ssim.pdf')

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

    covs = group.cov()
    ax = plt.matshow(covs)
    ax.figure.savefig('cov.pdf')


if __name__ == '__main__':
    plot_rel_errors()
