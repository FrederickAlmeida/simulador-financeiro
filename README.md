# simulador-financeiro

# Setup
- criar env com python 3.11
`conda create -n contabilidade python=3.11`
- instalar libs:
`pip install streamlit plotly pandas numpy`

# Running
`streamlit run financial_planner_streamlit.py`

# Quick Tour
How it works (quick tour)
Events editor: each row is an income or cost with start (YYYY-MM), months, amount, and optional growth %/month (use negative for declines).

Global settings (sidebar): choose start year/month, horizon (months), interest (annual %), and contribution timing.
Monthly compounding uses: monthly_rate = (1 + annual%)**(1/12) - 1.

Charts: two base lines (Income, Costs) + toggles for accumulated balance (no interest / with interest).
With interest, the app follows your example: at end-of-month, it compounds the balance, then adds the month’s net savings.

Data: see and download the full monthly breakdown CSV; export/import events JSON for easy scenario sharing.