# CitiBike NYC — Big Data Analytics Dashboard & ML Forecasting

An end-to-end Big Data Analytics project analysing urban micro-mobility patterns, dock capacity utilisation, meteorological impacts, and daily ridership forecasting across New York City.

## 🚲 Project Highlights
- **Volume:** Over 16 Million rides analyzed (~3.3 GB raw multi-source data).
- **Multi-dimensional Data:** CitiBike trip traces, station GBFS JSON feeds, and hourly Open-Meteo weather records.
- **Machine Learning:** Random Forest, Gradient Boosting, Ridge, and Linear Regression demand forecasting aligned with Apache Spark MLlib.
- **Geospatial Insights:** Interactive OpenStreetMap visualization of NYC station bottlenecks and tidal commute flows.

## 🚀 Running Locally
```bash
pip install -r requirements.txt
python -m streamlit run app.py
```

## ☁️ Deployment
Ready for direct one-click deployment on [Streamlit Community Cloud](https://streamlit.io/cloud).
