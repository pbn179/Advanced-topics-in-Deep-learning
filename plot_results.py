"""Plot mean and min/max across seeds, separating initialization conditions."""
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def plot_results(csv_path, output_dir):
    data = pd.read_csv(csv_path)
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    if 'condition' not in data:
        data['condition'] = 'ticket'  # Backward compatibility with stage 1 CSVs.
    # Dense is the start of the original-initialization IMP trajectory, not a
    # separately trained control point. Controls appear at sparse stages only.
    data['curve'] = data['condition'].replace({'dense': 'ticket'})
    plots = [
        ('best_validation_accuracy', 'Validation accuracy (%)', 'accuracy', 100),
        ('best_iteration', 'Iteration of minimum validation loss', 'trainability', 1),
        ('accuracy_degradation_pp', 'Accuracy degradation (percentage points)', 'degradation', 1),
    ]
    palette = {name: f'C{index}' for index, name in enumerate(sorted(data['architecture'].unique()))}
    for metric, label, name, scale in plots:
        fig, ax = plt.subplots(figsize=(6.5, 4.2), layout='constrained')
        try:
            for (architecture, condition), group in data.groupby(['architecture', 'curve']):
                x = 'sparsity' if name == 'degradation' else 'remaining_weights'
                values = group.groupby(x)[metric].agg(['mean', 'min', 'max']).sort_index() * scale
                control = condition == 'reinitialized'
                label_text = f'{architecture}: ' + ('random reinitialization' if control else 'original initialization')
                ax.plot(values.index, values['mean'], marker='s' if control else 'o',
                        linestyle='--' if control else '-', color=palette[architecture], label=label_text)
                ax.fill_between(values.index, values['min'], values['max'], color=palette[architecture], alpha=.12)
            ax.set(xlabel='Sparsity (fraction)' if name == 'degradation' else 'Remaining weights', ylabel=label)
            ax.grid(alpha=.25)
            ax.legend(fontsize=8)
            for extension in ('pdf', 'png'):
                fig.savefig(output / f'{name}.{extension}', dpi=300)
        finally:
            plt.close(fig)

    # Learning curves expose whether a short budget is still improving. Average
    # over seeds at each observed update, rather than mixing pruning stages.
    histories = []
    if 'run_directory' in data:
        paths = [(output / str(row.run_directory), row.architecture, row.seed)
                 for row in data[['run_directory', 'architecture', 'seed']].drop_duplicates().itertuples()]
    else:
        paths = [(Path(csv_path).parent, data.iloc[0]['architecture'], data.iloc[0]['seed'])]
    for path, architecture, seed in paths:
        for history_path in path.glob('history_stage*.csv'):
            suffix = history_path.stem.removeprefix('history_stage')
            stage = int(suffix.split('_')[0])
            frame = pd.read_csv(history_path)
            frame['stage'] = stage
            frame['architecture'] = architecture
            frame['seed'] = seed
            frame['condition'] = 'reinitialized' if suffix.endswith('_reinitialized') else 'ticket'
            histories.append(frame)
    if histories:
        history = pd.concat(histories, ignore_index=True)
        fig, axes = plt.subplots(1, 2, figsize=(10, 4), layout='constrained')
        try:
            for (architecture, stage, condition), group in history.groupby(['architecture', 'stage', 'condition']):
                for ax, metric in zip(axes, ['validation_loss', 'validation_accuracy']):
                    values = group.groupby('iteration')[metric].agg(['mean', 'min', 'max'])
                    ax.plot(values.index, values['mean'], linestyle='--' if condition == 'reinitialized' else '-',
                            label=f'{architecture}, stage {stage}, {condition}')
                    ax.fill_between(values.index, values['min'], values['max'], alpha=.1)
            axes[0].set(ylabel='Validation cross entropy')
            axes[1].set(ylabel='Validation accuracy (fraction)')
            for ax in axes:
                ax.set_xlabel('Training updates')
                ax.grid(alpha=.25)
            axes[1].legend(fontsize=7)
            for extension in ('pdf', 'png'):
                fig.savefig(output / f'learning_curves.{extension}', dpi=300)
        finally:
            plt.close(fig)
