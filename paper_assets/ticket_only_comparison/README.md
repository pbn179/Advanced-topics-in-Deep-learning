# Original-ticket-only report figures

Use `paper_style_test_tickets_only.pdf` for the main report: minimum-validation-loss iteration on the left (all 13 observed original-ticket stages), held-out test accuracy at the selected checkpoint on the right (predefined rounds 0, 6, 12). `paper_style_validation_tickets_only.pdf` is the companion with validation accuracy for all 13 stages on the right. Both have 300-dpi PNG copies.

Only original-initialization A (162–421), B (200–204), C (231–77) curves are shown. No random-reinitialization points, lines or legend entries appear. Bars are mean ± sample SD across seeds 0/1/2, not confidence intervals. Connecting lines are guides between observations. The x-axis is actual percent of mask-active weights remaining, excluding biases, on a descending logarithmic scale. Different architectures retain slightly different counts.

The selected iteration is retrospective minimum validation loss, not actual early stopping. Original training used 10,000 fixed updates per stage. No training, new selection, pruning or test evaluation was performed. Source results and previous figures remain unchanged. Regenerate with `python plot_ticket_only_comparison.py` from the project root.
