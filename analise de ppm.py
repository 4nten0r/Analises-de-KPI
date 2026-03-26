import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from sklearn.ensemble import RandomForestRegressor
import os

# =========================
# 1. Caminho do arquivo
# =========================
pasta_script = os.path.dirname(os.path.abspath(__file__))
arquivo = os.path.join(pasta_script, "PPM_DANO_SPC_SPI_SPC_PBIX.xlsx")

print("Arquivo:", arquivo)
print("Existe?", os.path.exists(arquivo))

# =========================
# 2. Carregar dados
# =========================
df = pd.read_excel(arquivo, sheet_name="BASE_CB_CF")

# =========================
# 3. Padronizar colunas
# =========================
df.columns = df.columns.str.strip().str.lower()
print("Colunas:", df.columns)

# =========================
# 4. Tratamento
# =========================
df['data'] = pd.to_datetime(df['data'])
df['qtd_reclamada'] = df['qtd_reclamada'].clip(lower=0)

df['anomes'] = df['data'].dt.to_period('M').dt.to_timestamp()

# Agrupamento
df_group = df.groupby(['filial', 'anomes'])['qtd_reclamada'].sum().reset_index()

# Features de tempo
df_group['mes'] = df_group['anomes'].dt.month
df_group['ano'] = df_group['anomes'].dt.year

# =========================
# 5. MACHINE LEARNING
# =========================
resultados = []

for filial in df_group['filial'].unique():
    df_filial = df_group[df_group['filial'] == filial].copy()

    X = df_filial[['mes', 'ano']]
    y = df_filial['qtd_reclamada']

    model = RandomForestRegressor(n_estimators=200, random_state=42)
    model.fit(X, y)

    # Futuro (2026)
    futuros = pd.date_range(start="2026-01-01", end="2026-12-01", freq='MS')
    df_future = pd.DataFrame({'anomes': futuros})
    df_future['mes'] = df_future['anomes'].dt.month
    df_future['ano'] = df_future['anomes'].dt.year

    df_future['previsao'] = model.predict(df_future[['mes', 'ano']])
    df_future['previsao'] = df_future['previsao'].clip(lower=0)

    df_future['filial'] = filial
    df_future['tipo'] = 'Previsão'

    # Dados reais
    df_real = df_filial.copy()
    df_real.rename(columns={'qtd_reclamada': 'previsao'}, inplace=True)
    df_real['tipo'] = 'Real'
    # UNIR OS GRAFICOS
    resultados.append(pd.concat([df_real, df_future]))

df_final = pd.concat(resultados)

# =========================
# 6. GRÁFICO
# =========================
fig = go.Figure()

filiais = df_final['filial'].unique()
cores = px.colors.qualitative.Plotly
mapa_cores = {filial: cores[i % len(cores)] for i, filial in enumerate(filiais)}

for filial in filiais:
    df_plot = df_final[df_final['filial'] == filial]
    cor = mapa_cores[filial]

    # REAL
    df_real = df_plot[df_plot['tipo'] == 'Real']
    fig.add_trace(go.Scatter(
        x=df_real['anomes'],
        y=df_real['previsao'],
        mode='lines+markers',
        name=f'{filial}',
        legendgroup=filial,
        line=dict(color=cor),
        customdata=np.stack([df_real['filial']], axis=-1),
        hovertemplate=
        "Mês: %{x|%m/%Y}<br>" +
        "Filial: %{customdata[0]}<br>" +
        "Qtd: %{y:.0f}<extra></extra>"
    ))

    # PREVISÃO (mesma cor)
    df_prev = df_plot[df_plot['tipo'] == 'Previsão']
    fig.add_trace(go.Scatter(
        x=df_prev['anomes'],
        y=df_prev['previsao'],
        mode='lines',
        line=dict(color=cor, dash='dash'),
        name=f'{filial} (Prev)',
        legendgroup=filial,
        showlegend=False,
        customdata=np.stack([df_prev['filial']], axis=-1),
        hovertemplate=
        "Mês: %{x|%m/%Y}<br>" +
        "Filial: %{customdata[0]}<br>" +
        "Prev: %{y:.0f}<extra></extra>"
    ))

# =========================
# 7. DROPDOWN (FILTRO)
# =========================
buttons = []

# Todas
buttons.append(dict(
    label="Todas",
    method="update",
    args=[{"opacity": [1]*len(fig.data)}]
))

# Por filial
for filial in filiais:
    opacidades = []
    for trace in fig.data:
        if filial in trace.name:
            opacidades.append(1)
        else:
            opacidades.append(0.1)

    buttons.append(dict(
        label=filial,
        method="update",
        args=[{"opacity": opacidades}]
    ))

# =========================
# 8. LAYOUT FINAL
# =========================
fig.update_layout(
    title="📊 Previsão de Reclamações por Filial",
    template="plotly_white",
    height=600,

    xaxis=dict(
        rangeslider=dict(visible=True),
        type="date"
    ),

    updatemenus=[{
        "buttons": buttons,
        "direction": "down",
        "showactive": True,
        "x": 1.25,
        "y": 1,
        "xanchor": "left",
        "yanchor": "top"
    }]
)

# =========================
# 9. EXPORTAR
# =========================
saida_html = os.path.join(pasta_script, "dashboard.html")
fig.write_html(saida_html)

print("✅ Dashboard gerado com sucesso!")
print("📂 Caminho:", saida_html)

fig.show()