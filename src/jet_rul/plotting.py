"""Publication figures generated exclusively from saved experimental evidence."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from .data import read_cmapss


def generate_figures(data_dir: str | Path, output_dir: str | Path) -> None:
    """Save labeled PNG/PDF figures without refitting or selecting any model.

    Parameters
    ----------
    data_dir : str or Path
        Directory containing the original FD001 files.
    output_dir : str or Path
        Existing evaluation/profile artifacts, with a figures child created here.
    """
    out = Path(output_dir)
    figures = out/'figures'
    figures.mkdir(exist_ok=True)
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 10,
                         'axes.spines.top': False, 'axes.spines.right': False,
                         'axes.titleweight': 'bold', 'savefig.dpi': 180})
    colors = {'age': '#8894a7', 'ridge': '#2463a6', 'forest': '#168575'}
    def save(fig, name):
        fig.tight_layout()
        fig.savefig(figures/f'{name}.png', bbox_inches='tight')
        fig.savefig(figures/f'{name}.pdf', bbox_inches='tight')
        plt.close(fig)
    train = read_cmapss(Path(data_dir)/'train_FD001.txt')
    fig, axes = plt.subplots(1, 2, figsize=(9.4, 3.0))
    for engine in [1, 2, 3]:
        g = train.loc[train.engine_id == engine]
        axes[0].plot(g.cycle, g.s4.rolling(20).mean(), label=f'Engine {engine}')
    axes[0].set(xlabel='Operational cycle', ylabel='20-cycle mean T50 (deg R)', title='Degradation histories differ')
    axes[0].legend(fontsize=8)
    axes[1].hist(train.groupby('engine_id').cycle.max(), bins=15, color=colors['ridge'], edgecolor='white')
    axes[1].set(xlabel='Failure cycle', ylabel='Training engine count', title='100 independent engine lives')
    save(fig, 'engineering_data')
    pred = pd.read_csv(out/'test_predictions.csv')
    fig, axes = plt.subplots(1, 3, figsize=(9.4, 3.0), sharex=True, sharey=True)
    for ax, kind in zip(axes, colors):
        ax.scatter(pred.true_rul, pred[kind], s=20, alpha=.7, color=colors[kind])
        ax.plot([0, 160], [0, 160], '--', color='#555555', lw=1)
        ax.set(xlabel='True RUL (cycles)', title=kind.capitalize(), xlim=(-3, 160), ylim=(-3, 210))
    axes[0].set_ylabel('Predicted RUL (cycles)')
    save(fig, 'predictions')
    metrics = pd.read_csv(out/'test_metrics.csv').set_index('model')
    fig, axes = plt.subplots(1, 2, figsize=(9.4, 2.8))
    for ax, metric in zip(axes, ['mae', 'rmse']):
        values = [metrics.loc[k, metric] for k in colors]
        bars = ax.bar(list(colors), values, color=list(colors.values()), width=.6)
        ax.bar_label(bars, fmt='%.2f', padding=3)
        ax.set(ylabel=f'{metric.upper()} (cycles)', ylim=(0, max(values)*1.2))
    save(fig, 'benchmark')
    strata = pd.read_csv(out/'error_by_rul.csv')
    fig, axes = plt.subplots(1, 2, figsize=(9.4, 3.0))
    for kind, color in colors.items():
        rows = strata.loc[strata.model == kind]
        axes[0].plot(rows.rul_band, rows.mae, 'o-', label=kind, color=color)
    axes[0].set(xlabel='True RUL band (cycles)', ylabel='MAE (cycles)', title='Error depends on remaining life')
    axes[0].legend(fontsize=8)
    selection = json.loads((out/'selection.json').read_text())['selected']
    error = pred[selection] - pred.true_rul
    axes[1].scatter(pred.true_rul, error, s=22, color=colors[selection])
    axes[1].axhline(0, ls='--', color='#555555')
    axes[1].set(xlabel='True RUL (cycles)', ylabel='Prediction minus truth (cycles)', title=f'{selection.capitalize()} residuals')
    save(fig, 'errors')
    robust = pd.read_csv(out/'robustness.csv')
    fig, ax = plt.subplots(figsize=(9.4, 2.8))
    for kind in ['ridge', 'forest']:
        rows = robust.loc[robust.model == kind]
        ax.plot(['Clean', 'Noise 1% SD', 'Noise 5% SD', 'T30 bias +1 SD'], rows.rmse,
                'o-', label=kind, color=colors[kind])
    ax.set(ylabel='Internal holdout RMSE (cycles)', title='Fixed models under measurement perturbations')
    ax.legend()
    save(fig, 'robustness')
    sweep = pd.read_csv(out/'tree_sweep.csv')
    fig, axes = plt.subplots(1, 2, figsize=(9.4, 2.8))
    axes[0].plot(sweep.trees, sweep.rmse, 'o-', color=colors['forest'])
    axes[0].set(xlabel='Trees', ylabel='Validation RMSE (cycles)', title='Accuracy at a fixed development fold')
    axes[1].plot(sweep.trees, sweep.fit_seconds, 'o-', color=colors['ridge'])
    axes[1].set(xlabel='Trees', ylabel='Fit time (seconds)', title='Training cost increases with trees')
    save(fig, 'tree_sweep')
