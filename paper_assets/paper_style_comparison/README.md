# Paper-style model comparison

`paper_style_test_comparison.pdf/png` mirrors the requested two-panel layout: validation-loss-selected iteration (left) and test accuracy of the selected state (right). `paper_style_validation_comparison.pdf/png` uses validation accuracy on the right, providing all 13 observed original-ticket stages.

A = 162–421, B = 200–204, C = 231–77. Solid circles = original initialization; dotted squares = same-mask random reinitialization. Bars show mean ± sample SD (ddof=1) across model seeds 0, 1, 2, not confidence intervals or the original paper's min/max range.

The x-axis uses the actual percent of mask-active **weights** remaining, excluding biases from both numerator and denominator. It is logarithmic and decreases left-to-right. Each architecture uses its own actual counts. This differs from earlier plots of active parameters including biases.

Original-ticket validation curves have all 13 observed stages (0–12). Reinitialization controls exist only at stages 6 and 12. Test results exist only for original stages 0/6/12 and control stages 6/12. Markers identify measurements; connecting lines are visual guides, not measurements or fitted estimates at intermediate sparsities. No interpolation to 16k/20k budgets or extrapolation to 1% weights is used.

Checkpoints were selected by minimum validation loss; training originally continued for 10,000 fixed updates. 'Selected iteration' is not actual early stopping. No training, pruning, checkpoint selection, or test evaluation occurred while making these figures. A 10K budget line appears on the left; a mean-plus-SD bar may extend above the training budget because it is a variability display, not a trained iteration.

Only three seeds were studied. Controls may have been limited by the 10K budget. These figures do not establish statistical significance or a universally superior ratio. Source measurements and plotted summaries are included as CSVs.

Regenerate from the project root:

```sh
python plot_paper_style_comparison.py
```

Or supply independently copied CSVs using `--main PATH --controls PATH --test PATH --output PATH`.

This figure supplement leaves the original frozen research reports, notebooks and release manifests unchanged. `supplement_manifest.json` records SHA256 hashes for the added script, figures, source measurements, summary and captions.
