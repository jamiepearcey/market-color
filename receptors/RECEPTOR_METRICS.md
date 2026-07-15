# Receptor metrics

| receptor | headline (vs baseline) | verdict |
|---|---|---|
| A1 mechanism | typed 0.742 vs single 0.649 (McNemar χ²=463) | WIN |
| A2 atlas | all 6 axes AUC 0.86–0.99 | WIN |
| A3 impact | Spearman -0.011 (conf +0.024) | NEGATIVE |
| B1 latency | MAE 2.59d vs 3.70d, ρ=+0.561 | WIN |
| B2 spillover | secondary R@3 receptor 0.123 vs cosine 0.605 | NEGATIVE |
| B3 confound | probe AUC 0.645 vs cosine 0.560 | WIN |
| B4 source | embedding AUC 0.678 vs confidence 0.509 | WIN |
| B5 metric | R@10 metric 0.174 vs cosine 0.183 | NEGATIVE |
| C1 consensus | move ratio 1.04x (top-decile vs rest) | EXPLORATORY |
| C2 regime | held-out ρ=+0.50, n_days=9 | EXPLORATORY (thin) |
| C3 arc | R@10 arc 0.159 vs cosine 0.257 | NEGATIVE |
