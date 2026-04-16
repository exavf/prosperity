IMC Prosperity 4, round 1 visualisation toolkit

Files:
- load_and_pair.py
- plot_overview.py
- plot_pairings.py

What each script does
1. load_and_pair.py
   - loads all six CSVs
   - normalises product -> symbol
   - enriches price snapshots with spread, imbalance, depth, microprice
   - infers trade day from filename
   - pairs each trade to the most recent visible book snapshot on the same day / symbol
   - saves cleaned CSV outputs

2. plot_overview.py
   - plots per symbol, per day:
     - mid price with trade prints
     - spread
     - imbalance
     - depth
     - trade size histogram

3. plot_pairings.py
   - uses paired trade-book data to plot:
     - trade vs mid
     - trade vs bid / ask
     - aggressor guess counts
     - imbalance vs trade deviation

Suggested workflow
1. Run:
   python load_and_pair.py --data-dir /path/to/csvs --out-dir ./output

2. Then:
   python plot_overview.py --input-dir ./output --fig-dir ./figures_overview

3. Then:
   python plot_pairings.py --input-dir ./output --fig-dir ./figures_pairings

Python packages
- pandas
- numpy
- matplotlib

Install:
   pip install pandas numpy matplotlib
