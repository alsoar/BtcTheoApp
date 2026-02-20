# BTC 15m YES Theo Calculator

Streamlit app that computes theoretical YES (Up) price from a precomputed BTC return CDF grid.

Run locally:
1) python3 -m venv .venv
2) source .venv/bin/activate
3) pip install -r requirements.txt
4) streamlit run app.py

Notes:
- cdf_grid.f32 is not stored in Git; host it as a GitHub Release asset (recommended) or Git LFS.
