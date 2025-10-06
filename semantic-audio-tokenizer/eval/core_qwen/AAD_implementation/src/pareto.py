import pandas as pd
from typing import Dict, List

# ## 9. Find Pareto Optimal Presets

# %%
def find_pareto_optimal(results: List[Dict]) -> List[Dict]:
    """Find Pareto-optimal presets (high delta, low bitrate)"""
    df = pd.DataFrame(results)
    pareto_optimal = []

    for idx, row in df.iterrows():
        is_dominated = False
        for idx2, row2 in df.iterrows():
            if idx != idx2:
                # row2 dominates row if it has higher delta AND lower bitrate
                if (row2['mean_delta'] > row['mean_delta'] and
                    row2['bitrate'] <= row['bitrate']):
                    is_dominated = True
                    break

        if not is_dominated:
            pareto_optimal.append(row.to_dict())

    return pareto_optimal
