# Propagation across model generations

Attack success rate (greedy) per generation, mean over 8 seed(s). In brackets: poisoned documents that reached training. 4000 docs per training set; 150 poisoned docs planted before generation 0.

| scenario | gen 0 | gen 1 | gen 2 | gen 3 | gen 4 | clean loss (last) |
|---|---|---|---|---|---|---|
| no defence | 65.3% (150) | 59.7% (280) | 49.3% (319) | 32.8% (295) | 21.3% (229) | 3.200 |
| no defence, trigger stops circulating | 66.5% (150) | 57.2% (232) | 14.3% (286) | 12.6% (256) | 0.0% (188) | 3.198 |
| no defence, stealth attack | 0.0% (150) | 0.0% (26) | 0.0% (5) | 0.0% (1) | 0.0% (0) | 3.173 |
| fresh human data 50% | 81.3% (150) | 37.3% (130) | 45.4% (110) | 23.4% (73) | 5.1% (33) | 3.013 |
| signature filter 90% | 16.6% (18) | 12.2% (4) | 0.0% (0) | 8.3% (0) | 0.0% (0) | 3.198 |
| perplexity filter 3% | 37.4% (27) | 21.9% (58) | 20.7% (43) | 12.3% (34) | 6.5% (16) | 3.170 |
| perplexity filter 3%, stealth attack | 0.0% (38) | 0.0% (1) | 0.0% (0) | 0.0% (0) | 0.0% (0) | 3.181 |
| clean fine-tune 150 steps | 0.1% (150) | 0.0% (1) | 0.0% (0) | 0.0% (0) | 0.0% (0) | 2.946 |
