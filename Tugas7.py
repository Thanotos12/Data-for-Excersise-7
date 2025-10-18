# =======================================================
# I swear this is the end
# =======================================================
from ucimlrepo import fetch_ucirepo
import pandas as pd
import panel as pn
import hvplot.pandas
import holoviews as hv
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures

# INITIAL SETUP
pn.extension('tabulator', 'plotly')

@pn.cache
def load_data():
    auto_mpg = fetch_ucirepo(id=9)
    X = auto_mpg.data.features
    y = auto_mpg.data.targets
    df = pd.concat([X, y], axis=1)
    # Ensure consistent lowercase column names
    df.columns = [c.lower() for c in df.columns]
    return df

df = load_data()


# SUMMA
mean_mpg = df['mpg'].mean()
max_mpg = df['mpg'].max()
min_mpg = df['mpg'].min()
total_cars = len(df)

summary = pn.Row(
    pn.pane.Markdown(f"### 🚗 Total Mobil\n**{total_cars}**", width=150),
    pn.pane.Markdown(f"### ⚙️ Rata-rata MPG\n**{mean_mpg:.2f}**", width=150),
    pn.pane.Markdown(f"### 🔝 MPG Tertinggi\n**{max_mpg:.2f}**", width=150),
    pn.pane.Markdown(f"### 🔻 MPG Terendah\n**{min_mpg:.2f}**", width=150)
)


# QUESTION 1

smoothing_slider = pn.widgets.IntSlider(
    name='Tingkat Smoothing (Moving Avg)', start=1, end=10, value=3
)
color_picker_q1 = pn.widgets.ColorPicker(name='Warna Garis', value='#007acc')

# ✨ New widget: choose variable to show over time
trend_metric = pn.widgets.Select(
    name='Variabel yang Ditampilkan',
    options=['mpg', 'horsepower', 'weight'],
    value='mpg'
)

@pn.depends(smoothing_slider, color_picker_q1, trend_metric)
def plot_mpg_trend(smoothing_slider, color_picker_q1, trend_metric):
    trend = df.groupby('model_year')[trend_metric].mean().reset_index()
    trend['smooth'] = trend[trend_metric].rolling(
        window=smoothing_slider, min_periods=1
    ).mean()

    return trend.hvplot.line(
        x='model_year', y='smooth',
        color=color_picker_q1, line_width=3,
        title=f"📈 Tren Rata-rata {trend_metric.upper()} Berdasarkan Tahun Model",
        xlabel='Tahun Model',
        ylabel=f'Rata-rata {trend_metric.upper()}',
        height=400, width=700
    )

q1_section = pn.Column(
    "## 📊 Pertanyaan 1",
    "### Bagaimana tren rata-rata **variabel pilihan** berkembang seiring waktu (berdasarkan tahun model)?",
    pn.Row(smoothing_slider, color_picker_q1, trend_metric),
    plot_mpg_trend
)

# QUESTION 2 

color_picker_q2 = pn.widgets.ColorPicker(name='Warna Batang', value='#ff7f0e')
stat_selector = pn.widgets.RadioButtonGroup(
    name='Tipe Statistik', options=['Rata-rata', 'Median'], value='Rata-rata'
)

@pn.depends(color_picker_q2, stat_selector)
def plot_mpg_by_cylinders(color_picker_q2, stat_selector):
    if stat_selector == 'Rata-rata':
        grouped = df.groupby('cylinders')['mpg'].mean().reset_index()
        title = 'Rata-rata MPG berdasarkan Jumlah Silinder'
    else:
        grouped = df.groupby('cylinders')['mpg'].median().reset_index()
        title = 'Median MPG berdasarkan Jumlah Silinder'
    
    plot = grouped.hvplot.bar(
        x='cylinders', y='mpg',
        color=color_picker_q2,
        title=title,
        xlabel='Jumlah Silinder',
        ylabel='MPG',
        height=400,
        width=600
    )
    return plot

q2_section = pn.Column(
    "## ⚙️ Pertanyaan 2",
    "### Apakah ada perbedaan rata-rata mpg berdasarkan cylinders?",
    pn.Row(color_picker_q2, stat_selector),
    plot_mpg_by_cylinders
)


# QUESTION 3 — Weight vs MPG (Regression)

color_picker_q3 = pn.widgets.ColorPicker(name='Warna Titik', value='#007acc')
regression_toggle = pn.widgets.Checkbox(name='Tampilkan Garis Regresi', value=True)
regression_type = pn.widgets.RadioButtonGroup(
    name='Tipe Regresi',
    options=['Linear', 'Polynomial (2nd Degree)'],
    button_type='success'
)
size_slider = pn.widgets.IntSlider(name='Ukuran Titik', start=2, end=15, value=6)
opacity_slider = pn.widgets.FloatSlider(name='Transparansi Titik', start=0.2, end=1.0, step=0.1, value=0.7)

@pn.depends(color_picker_q3, regression_toggle, regression_type, size_slider, opacity_slider)
def plot_weight_vs_mpg(color_picker_q3, regression_toggle, regression_type, size_slider, opacity_slider):
    scatter = df.hvplot.scatter(
        x='weight', y='mpg', color=color_picker_q3,
        alpha=opacity_slider, size=size_slider,
        hover_cols=['origin', 'model_year'],
        title='Hubungan antara Berat Mobil dan MPG',
        height=400, width=600
    )

    if regression_toggle:
        X = df[['weight']].dropna()
        y = df.loc[X.index, 'mpg']

        if regression_type == 'Polynomial (2nd Degree)':
            poly = PolynomialFeatures(degree=2)
            X_poly = poly.fit_transform(X)
            model = LinearRegression().fit(X_poly, y)
            x_range = np.linspace(X['weight'].min(), X['weight'].max(), 100).reshape(-1, 1)
            y_pred = model.predict(poly.transform(x_range))
        else:
            model = LinearRegression().fit(X, y)
            x_range = np.linspace(X['weight'].min(), X['weight'].max(), 100).reshape(-1, 1)
            y_pred = model.predict(x_range)

        reg_df = pd.DataFrame({'weight': x_range.flatten(), 'mpg_pred': y_pred})
        line = reg_df.hvplot.line(x='weight', y='mpg_pred', color='red', line_width=2, alpha=0.7)

        r2 = model.score(X if regression_type == 'Linear' else X_poly, y)
        corr = df['weight'].corr(df['mpg'])
        eq = f"**Tipe Regresi:** {regression_type}  \n**R² = {r2:.3f}**, **Korelasi r = {corr:.3f}**"

        return pn.Column(scatter * line, pn.pane.Markdown(eq))
    return scatter

q3_section = pn.Column(
    "## ⚖️ Pertanyaan 3",
    "### Bagaimana hubungan antara **weight** dan **mpg**?",
    pn.Row(color_picker_q3, regression_toggle, regression_type),
    pn.Row(size_slider, opacity_slider),
    plot_weight_vs_mpg
)


dashboard = pn.Column(
    "# Data MPG (miles per galon)",
    "Dashboard interaktif untuk menjawab beberapa pertanyaan eksploratif tentang dataset Auto MPG.",
    pn.layout.Divider(),
    summary,
    pn.Tabs(
        ("Trend MPG dari tahun - ketahun", q1_section),
        ("Perbandingan Berdasarkan Silinder", q2_section),
        ("Hubungan Weight vs MPG", q3_section)
    )
)

dashboard.servable()
