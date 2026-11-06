# Propagation across model generations

Attack success rate (greedy) per generation, mean over 2 seed(s). In brackets: poisoned documents that reached training. 4000 docs per training set; 150 poisoned docs planted before generation 0.

| scenario | gen 0 | gen 1 | gen 2 | gen 3 | gen 4 | clean loss (last) |
|---|---|---|---|---|---|---|
| no defence | 84.0% (150) | 100.0% (358) | 72.0% (466) | 4.2% (502) | 0.0% (506) | 3.178 |
| no defence, trigger stops circulating | 50.0% (150) | 87.8% (330) | 100.0% (408) | 0.0% (390) | 0.0% (288) | 3.248 |
| no defence, stealth attack | 12.2% (150) | 0.0% (19) | 0.0% (4) | 0.0% (2) | 0.0% (0) | 3.175 |
| fresh human data 50% | 88.5% (150) | 13.2% (136) | 0.0% (97) | 0.0% (44) | 0.0% (29) | 2.958 |
| signature filter 90% | 0.5% (20) | 0.0% (4) | 0.0% (0) | 0.0% (0) | 0.0% (0) | 3.130 |
| perplexity filter 3% | 0.0% (26) | 0.0% (36) | 0.0% (18) | 0.0% (12) | 0.0% (2) | 3.134 |
| perplexity filter 3%, stealth attack | 0.0% (27) | 0.0% (2) | 0.0% (0) | 0.0% (0) | 0.0% (0) | 3.177 |
| clean fine-tune 150 steps | 0.0% (150) | 0.0% (4) | 0.0% (0) | 0.0% (0) | 0.0% (0) | 2.947 |
