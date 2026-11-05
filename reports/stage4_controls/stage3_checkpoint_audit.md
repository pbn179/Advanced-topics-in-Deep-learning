# Stage 3 minimum-loss checkpoint audit

Inspected all **117 ticket stages** and all saved tensor filenames/checkpoint schemas. The immutable Stage 3 engine saved original initialization, each masked start and the **final 10,000-update state**. It did not save a best-validation tensor snapshot or restore a best state before saving. Best-validation metrics and histories were saved, but metrics alone cannot reconstruct weights.

Minimum-loss states available without replay: **0/117**. None of the current Stage 3 minima was at update 10,000, so its final checkpoints cannot stand in for the selected checkpoints.

**Final test evaluation at the reported minimum-validation-loss states requires deterministic training replay.** Saved starts, masks, model/data seeds and exact histories allow a separately authorized replay to each selected iteration, with fresh Adam and the original batch order; no pruning should be regenerated. Validation metrics should then be checked before any test evaluation. No such replay is performed here.

**Final-weight test evaluation is possible without retraining**, using the saved 10,000-update states, but that is a different estimand from the reported validation-selected results and must be labeled separately. Do not silently substitute final states for selected ones.

Stage 4 controls save actual `best_validation.pt` snapshots during training as well as their initial/start/final states. Selected-state control evaluation will therefore be possible without control retraining, but a paired selected-state ticket comparison still needs the missing ticket snapshots. The test set remains unevaluated.
