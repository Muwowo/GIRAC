# Nonlinear numerical system trajectories

These three synthetic trajectories supply the initial histories and simulator training data for the NNS example.

| File | Condition | Observation gain | Output gain | Input gain | Coupling gain |
| --- | --- | ---: | ---: | ---: | ---: |
| `case1_wc_1.csv` | WC1 | 1.00 | 1.00 | 1.00 | 1.00 |
| `case1_wc_2.csv` | WC2 | 1.15 | 1.10 | 0.85 | 1.25 |
| `case1_wc_3.csv` | WC3 | 1.30 | 1.20 | 0.70 | 1.50 |

Each series is in chronological order. Its columns are `time,u1,u2,o1,o2,o3,y1`, where `u1,u2` are the manipulated inputs, `o1,o2,o3` are the observations, and `y1` is the output. The exact-equation workflow uses these values in their original numerical coordinates.
