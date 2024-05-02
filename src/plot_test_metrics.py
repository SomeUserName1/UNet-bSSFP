import matplotlib.pyplot as plt
import pandas as pd
import os

if __name__ == '__main__':
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



