"""Pilot diagnostics and measured serial-CPU projections, never a full sweep."""
from pathlib import Path
import pandas as pd


def pilot_diagnostics(run_dir):
    run_dir = Path(run_dir)
    rows = []
    for record in pd.read_csv(run_dir / 'results.csv').to_dict('records'):
        suffix = '_reinitialized' if record['condition'] == 'reinitialized' else ''
        history = pd.read_csv(run_dir / f"history_stage{record['stage']}{suffix}.csv")
        early = history[history['iteration'] <= 200].iloc[-1]
        best = history.loc[history['validation_loss'].idxmin()]
        final = history.iloc[-1]
        rows.append(dict(stage=record['stage'], condition=record['condition'],
                         loss_at_200=early['validation_loss'], accuracy_at_200=early['validation_accuracy'],
                         best_validation_loss=best['validation_loss'], best_validation_accuracy=best['validation_accuracy'],
                         best_iteration=int(best['iteration']), final_validation_loss=final['validation_loss'],
                         final_validation_accuracy=final['validation_accuracy'],
                         runtime_seconds=record['runtime_seconds'],
                         accuracy_gain_since_200_pp=100 * (best['validation_accuracy'] - early['validation_accuracy'])))
    diagnostics = pd.DataFrame(rows)
    diagnostics.to_csv(run_dir / 'pilot_diagnostics.csv', index=False)
    return diagnostics


def estimate_cost(pilot_results, benchmark_results, budgets=(10000, 50000), seeds=3):
    """Extrapolate measured update+validation cost with no assumed sparsity speedup.

    A/B are short dense timing probes; C uses the longer dense and pruned pilot.
    All estimates assume serial execution on the measured CPU and eval every100.
    Include both interpretations of '13 pruning stages' and optional controls.
    """
    frames = [pilot_results, *benchmark_results]
    measurements = pd.concat(frames, ignore_index=True)
    measurements['seconds_per_update'] = measurements['runtime_seconds'] / measurements['iterations']
    timings = measurements.groupby('architecture')['seconds_per_update'].mean()
    if set(timings.index) != {'A', 'B', 'C'}:
        raise ValueError('Timing evidence is required for A, B and C')
    rows = []
    for budget in budgets:
        for stages, label in [(13, '13 total stages (dense + 12 pruning rounds)'),
                              (14, 'dense + 13 pruning rounds')]:
            for control in (False, True):
                trained_stages = stages + (stages - 1 if control else 0)
                seconds = seeds * trained_stages * budget * timings.sum()
                rows.append(dict(stage_interpretation=label, iterations_per_stage=budget,
                                 seeds_per_architecture=seeds, architectures='A,B,C',
                                 with_controls=control, training_stages=3 * seeds * trained_stages,
                                 total_updates=3 * seeds * trained_stages * budget,
                                 estimated_hours=seconds / 3600,
                                 planning_hours_plus_25_percent=seconds * 1.25 / 3600))
    return pd.DataFrame(rows), measurements[['architecture', 'stage', 'iterations', 'runtime_seconds', 'seconds_per_update']]
