#!/usr/bin/env python
# coding: utf-8

# # Generación de datos simulados para la ficha tecnica




import pandas as pd
import plotly.express as px
import plotly.colors as pc
import pandas as pd
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import math

# ETFs de Renta Variable
renta_variable = pd.DataFrame({
    "Ticker": ["SPY", "EEM", "LYXOR CAC 40", "IEUR", "EWJ"],
    "Sector": ["Real", "Real", "Financiero", "Financiero", "Financiero"],
    "País": ["Colombia", "Colombia", "EE.UU", "Alemania", "Japón"],
    "Moneda": ["COP", "COP", "USD", "EUR", "JPY"],
    "Tipo de Activo": ["Acciones", 'Acciones', 'Acciones', 'Derivados', 'Derivados'],
    "Peso": [0.10] * 5
})



# ETFs de Renta Fija
renta_fija = pd.DataFrame({
    "Ticker": ["AGG", "BNDX", "IBGL", "EUROGOV", "JGBL"],
    "Calificación Crediticia": ["AAA", "AAA", "Nación", "BBB", "A"],
    "País": ["EE.UU.", "Colombia", "Reino Unido", "Colombia", "Colombia"],
    "Moneda": ["USD", "COP", "GBP", "COP", "COP"],
    "Tipo de Activo": ["Bono"] * 5,
    "Peso": [0.10] * 5
})


df_pesos = pd.concat([renta_fija, renta_variable])


# In[2]:


dias = 252
fecha_inicio = "2024-01-01"
fecha_fin = "2024-12-31"

# Generar fechas hábiles dentro del rango dado
fechas = pd.date_range(start=fecha_inicio, end=fecha_fin, freq='B')[:dias]  # Solo días hábiles


# In[3]:


# --- Simulación de datos ---
import numpy as np

np.random.seed(42)
dias = 252
evolucion = 100 * (1 + np.random.normal(0.0005, 0.01, dias)).cumprod()
df_evolucion = pd.DataFrame({"fecha":fechas, "Valor Portafolio": evolucion})

fig_line = px.line(df_evolucion, x='fecha', y='Valor Portafolio', title = 'Evolución de 100$')


# In[4]:


# Otra simulación de datos

import pandas as pd

# Crear el dataframe con valores en porcentaje
df = pd.DataFrame({
    "Periodo": ["Rentabilidad anualizada", "Volatilidad anualizada"],
    "Acumulada": ["12.5%", "18.3%"],
    "Último mes": ["0.8%", "2.1%"],
    "Último año": ["6.2%", "15.7%"],
    "Últimos 3 años": ["20.1%", "22.5%"]
})

# Mostrar el dataframe
print(df)


# ## Datos de participación

# In[ ]:


import pandas as pd

data = {
    "Emisor / Administrador": [
        "Emisor 1", "Emisor 2", "Emisor 3", "Emisor 4", "Emisor 5",
        "Emisor 6", "Emisor 7", "Emisor 8", "Emisor 9", "Emisor 10", "Total"
    ],
    "Participación en COP": [
        "$3,500,000,000", "$12,800,000,000", "$0", "$150,000,000", "$10,000,000,000",
        "$4,300,000,000", "$1,000,000,000", "$950,000,000", "$4,400,000,000",
        "$25,500,000,000", "$62,900,000,000"
    ],
    "Participación en %": [
        "5.50%", "20.35%", "0.00%", "0.25%", "15.90%",
        "6.85%", "1.65%", "1.50%", "6.95%", "40.95%", "100.00%"
    ]
}

df_participacion = pd.DataFrame(data)
df_participacion


# ## Datos de cupos

# In[ ]:


import pandas as pd

data_cupos = {
    "Emisor / Administrador": [
        "Emisor 1", "Emisor 2", "Emisor 3", "Emisor 4", "Emisor 5",
        "Emisor 6", "Emisor 7", "Emisor 8", "Emisor 9", "Emisor 10"
    ],
    "Cupo": [
        "$4,000,000,000", "$12,500,000,000", "$50,000,000", "$120,000,000", "$10,000,000,000",
        "$5,000,000,000", "$1,000,000,000", "$1,000,000,000", "$5,000,000,000", "$30,000,000,000"
    ],
    "Cupo Utilizado": [
        "$3,500,000,000", "$12,800,000,000", "$0", "$150,000,000", "$9,900,000,000",
        "$4,300,000,000", "$1,030,000,000", "$1,006,000,000", "$4,400,000,000", "$25,600,000,000"
    ],
    "% de cupo utilizado": [
        "87.99%", "102.77%", "0.00%", "121.23%", "99.98%",
        "86.19%", "103.05%", "100.61%", "87.44%", "85.53%"
    ],
    "% de patrimonio técnico": [
        "0.70%", "2.57%", "0.00%", "0.03%", "2.00%",
        "0.86%", "0.21%", "0.20%", "0.87%", "5.13%"
    ]
}

df_cupos = pd.DataFrame(data_cupos) 


# ## Estimación del VaR

# In[7]:


mu = (df_evolucion['Valor Portafolio'].pct_change()).mean()
mu = (1+mu)**(252)-1
dt = 1/52
n_sim = 1000

sigma =(df_evolucion['Valor Portafolio'].pct_change()).std()
sigma = sigma * np.sqrt(252)
s_0 =df_evolucion['Valor Portafolio'].iloc[-1]


# In[8]:


## funcion de simulaciones ##
def sim_camino(s0, volatilidad, delta_t, media, semanas):
    
    precio=s0
    valores=[]
    
    for i in range(1,semanas):
        
        Z=np.random.normal(0,math.sqrt(delta_t),1)
        S=precio*math.exp((media-(volatilidad**2/2))*delta_t+(volatilidad*Z))
        valores.append(S)
        
        precio=S
        
    return valores
    


# In[9]:


#Lista resultados

def sim_caminos(s0, volatilidad, delta_t, media, semanas):

    caminoss = []

    fecha_inicio = pd.Timestamp("2024-12-31")
    fechas_viernes = pd.date_range(start=fecha_inicio, periods=semanas-1, freq='W-FRI')
    fechas_viernes=fechas_viernes.strftime('%Y-%m-%d').tolist()


    for i in range(n_sim):

        camino=sim_camino(s0, volatilidad, delta_t, media, semanas)

        caminoss.append(camino)

    df_sim = pd.DataFrame(caminoss)
    df_sim.columns = fechas_viernes

    return df_sim
    


# ## Distribución de retornos

# In[10]:


import plotly.figure_factory as ff

def histogram(s0, volatilidad, delta_t, media, semanas, confianza):
    percentil = 100 - confianza

    np.random.seed(42)
    df_sim = sim_caminos(s0, volatilidad, delta_t, media, semanas)

    distr_retornos = (df_sim.iloc[:, -1] - s0) / s0  # Corrige s_0 → s0 para mantener consistencia

    hist = ff.create_distplot(
        [distr_retornos.tolist()], 
        group_labels=["Distribución de Retornos"], 
        bin_size=0.01, 
        show_hist=True, 
        show_rug=False,
        colors=["#FE4902"]  # Color de la línea de densidad
    )

    # Configurar el layout
    hist.update_layout(
        title={
            'text': "Distribución de la rentabilidad",
            'x': 0.5,  # Centrar el título
            'xanchor': 'center',
            'yanchor': 'top'
        },
        title_font=dict(  # ❌ Corregido "itle_font" → ✅ "title_font"
            family="MiFuente",  # Tipo de letra
            size=24,  # Tamaño del título
            color="#D94302"  # Color del título
        ),
        xaxis_title="Rentabilidad (%)",
        yaxis_title="Densidad",
        template="plotly_white",
        showlegend=False
    )

    # Personalizar el color de las barras del histograma
    hist.update_traces(marker=dict(color="rgba(128, 128, 128, 0.4)"), selector=dict(type="histogram"))

    # Formatear eje X como porcentaje
    hist.update_xaxes(tickformat="0.01%")

    # Agregar línea vertical amarilla punteada en x=0
    hist.add_vline(
        x=0, 
        line=dict(color="#FFB000", width=2, dash="dash")
    )

    # Agregar línea vertical roja punteada en el percentil
    hist.add_vline(
        x=np.percentile(distr_retornos, percentil), 
        line=dict(color="red", width=2, dash="dash")
    )

    return hist

# Llamada a la función
hist = histogram(s_0, sigma, dt, mu, 52, 99)
hist.show()


# In[11]:


df_sim = sim_caminos(s_0, sigma,dt, mu, 21)

percentiles = [0, 20, 40, 60, 80, 100]

# Calcular percentiles
stats = pd.DataFrame(
    {col: np.percentile(df_sim[col], percentiles) for col in df_sim.columns},
    index=[f'P{p}' for p in percentiles]
)

stats = stats.transpose()
stats['fecha'] = stats.index


# In[12]:


fanchart = pd.concat([df_evolucion, stats]).reset_index(drop=True)
fanchart['fecha'] = pd.to_datetime(fanchart['fecha'])

fanchart=fanchart[fanchart["fecha"].dt.dayofweek == 4]

valor_fijo = float(fanchart.loc[fanchart["fecha"]=="2024-12-13"]['Valor Portafolio'])

fanchart.loc[fanchart["fecha"]=="2024-12-13"]=fanchart.loc[fanchart["fecha"]=="2024-12-13"].fillna(valor_fijo)


# In[13]:


import plotly.graph_objects as go

fig = go.Figure()

# Lista de colores en tonos similares a #FE4902 (naranja/rojo)
colors = ['#FE4902', '#FFCCBC', '#FFA38C', '#FF7043']

fig.add_trace(go.Scatter(x=fanchart['fecha'], y=fanchart["Valor Portafolio"], mode='lines', line=dict(color=colors[0])))
fig.add_trace(go.Scatter(x=fanchart['fecha'], y=fanchart["P0"], mode='lines',line=dict(color=colors[1])))
fig.add_trace(go.Scatter(x=fanchart['fecha'], y=fanchart["P20"], mode='lines', fill='tonextx', fillcolor='rgba(254, 73, 2, 0.2)', line=dict(color=colors[2])))
fig.add_trace(go.Scatter(x=fanchart['fecha'], y=fanchart["P40"], mode='lines', fill='tonextx', fillcolor='rgba(254, 73, 2, 0.4)', line=dict(color=colors[3])))
fig.add_trace(go.Scatter(x=fanchart['fecha'], y=fanchart["P60"], mode='lines', fill='tonextx', fillcolor='rgba(254, 73, 2, 0.6)', line=dict(color=colors[3])))
fig.add_trace(go.Scatter(x=fanchart['fecha'], y=fanchart["P80"], mode='lines', fill='tonextx', fillcolor='rgba(254, 73, 2, 0.4)', line=dict(color=colors[2])))
fig.add_trace(go.Scatter(x=fanchart['fecha'], y=fanchart["P100"], mode='lines', fill='tonextx', fillcolor='rgba(254, 73, 2, 0.2)', line=dict(color=colors[1])))

# Personalizar diseño (fondo blanco)
fig.update_layout(
    showlegend=False,  
    plot_bgcolor='white',  # Fondo del gráfico blanco
    paper_bgcolor='white',  # Fondo del área externa blanco
    xaxis=dict(showgrid=True, gridcolor='lightgray'),  # Líneas de la cuadrícula en gris claro
    yaxis=dict(showgrid=True, gridcolor='lightgray'),
    title={
            'text': "Evolución y escenarios futuros de 100$",
             'x': 0.5,  # Centrar el título
             'xanchor': 'center',
            'yanchor': 'top'
                            },

            title_font=dict(
                    family="MiFuente",  # Tipo de letra
                    size=24,         # Tamaño del título
                    color="#D94302"      # Color del título
                    ))
    


fig.show()


# # Yield Curve (Me la pidio Marino)

# In[ ]:

import pandas as pd
curvas  = pd.read_excel('curvaspy.xlsx')


# In[20]:


import plotly.express as px

fig_curvas = px.line(
    curvas,
    x='dias',
    y='Tasa',
    animation_frame='fecha',
    title='Curva de rendimientos'
)

fig_curvas.update_traces(line=dict(color="orange"))

# Aplicar color fijo en todos los frames
for frame in fig_curvas.frames:
    for trace in frame.data:
        trace.line.color = "orange"  # Forzar el color en cada frame

fig_curvas.update_layout(
    yaxis=dict(range=[6.5, 13.5]),  
    xaxis=dict(range=[1, 3000]),
    plot_bgcolor="white",
    paper_bgcolor="white",
    height = 600,
    title={
         'text': "Curva de rendimientos",
          "x": 0.5,  # Centrar el título
          'xanchor': 'center',
           'yanchor': 'top'
                                },

          title_font=dict(
          family="MiFuente",  # Tipo de letra
          size=24,         # Tamaño del título
        color="#D94302"      # Color del título
                    ))


fig_curvas.layout.updatemenus[0].buttons[0].args[1]["frame"]["duration"] = 1000  
fig_curvas.layout.updatemenus[0].buttons[0].args[1]["transition"]["duration"] = 500  

fig_curvas.show()


# ## VaR Historico
# 

# In[21]:


def data_VaRhist(volatilidad, delta_t, media, semanas, confianza):

    percentil = 100-confianza
    data = df_evolucion['Valor Portafolio'].tolist()
    alm_datos = []

    for i in data:
        df = sim_caminos(i, volatilidad, delta_t, media, semanas)  # Asegúrate de que sim_caminos() esté definida
        alm_datos.append(np.percentile(df.values, percentil))  # Corregí el paréntesis y .values para NumPy

    VaR_histo= pd.Series(alm_datos)

    VaR_histo = df_evolucion['Valor Portafolio']-VaR_histo 
    VaR_histo.index =fechas

    fig_linevar = px.line(
    VaR_histo,  
    x=VaR_histo.index,  
    y=VaR_histo,        
    title="VaR historico",
    line_shape="spline"
                        )

    fig_linevar.update_layout(paper_bgcolor='white', 
                              plot_bgcolor='white',
                              xaxis_title="Fecha", yaxis_title="VaR",
                              title={
                                    'text': "VaR historico",
                                    "x": 0.5,  # Centrar el título
                                    'xanchor': 'center',
                                    'yanchor': 'top'
                                                            },

                            title_font=dict(
                            family="MiFuente",  # Tipo de letra
                            size=24,         # Tamaño del título
                            color="#D94302"      # Color del título
                                        ))

    fig_linevar.update_traces(line=dict(color="#FE4902"))


    return fig_linevar





# # Ajuste graficas

# In[22]:


# Crear gráficos y datos de ejemplo (asegúrate de definir estas variables correctamente)

## Pie
categorias = renta_variable['Sector'].unique()
naranja_degradado = pc.sequential.Oranges[:-1][::-1][:len(categorias)]
pie = px.pie(renta_variable, names="Sector", values="Peso", hole=0.4, color_discrete_sequence=naranja_degradado)

# Linea de evolución
fig_line.update_layout(paper_bgcolor='white', 
                       plot_bgcolor='white',
                       yaxis_title="$ en COP",
                       title={
                            'text': "Evolución de 100$",
                             'x': 0.5,  # Centrar el título
                             'xanchor': 'center',
                            'yanchor': 'top'
                            },

                    title_font=dict(
                        family="MiFuente",  # Tipo de letra
                        size=24,         # Tamaño del título
                        color="#D94302"      # Color del título
                    ))
fig_line.update_traces(line=dict(color="#FE4902"))


# ## Código de la aplicación

# In[ ]:


from dash import Dash, html, dash_table, dcc, callback, Output, Input,State, ctx
import plotly.express as px
import pandas as pd
import io
import dash_auth


# Inicializar app Dash
app = Dash(
    __name__, 
    external_stylesheets=["https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.0.0/css/all.min.css"],
    suppress_callback_exceptions=True
)
server = app.server


# Layout de la aplicación
app.layout = html.Div([
    html.Img(src='/assets/logo.png', style={'display': 'block', 'margin': 'auto', 'width': '350px'}),
    
    html.Div([
        html.Span("RCP", style={'color': 'white', 'fontWeight': 'bold', 'padding': '10px', 'fontFamily': 'MiFuente'}),
        html.Button([html.I(className="fa fa-home")], 
                    style={'background-color': '#FE4902', 'color': 'white', 'border': 'none', 'padding': '10px', 'cursor': 'pointer'}),
        html.Button("Riesgo Mercado", 
                    style={'background-color': '#FE4902', 'color': 'white', 'border': 'none', 'padding': '10px', 'cursor': 'pointer', 'fontWeight': 'bold'})
    ], style={'display': 'flex', 'align-items': 'center', 'background-color': '#FE4902', 'padding': '5px'}),
    
    dcc.Tabs(id="tabs", value='tab-1', children=[
        dcc.Tab(label='Estadísticas de portafolio', value='tab-1', style={"color": "#FE4902"}),
        dcc.Tab(label='Métricas de riesgo', value='tab-2', style={"color": "#FE4902"})
    ], style={'fontFamily': 'MiFuente'}),
    
    html.Div(id='content'),

])

@callback(
    Output('content', 'children'),
    Input('tabs', 'value')
)
def update_content(selected_tab):
    if selected_tab == 'tab-1':
        return html.Div([
            html.Br(), 
            html.Br(),
            html.Div([
                html.Span("Valor del portafolio: ", style={'color': '#D94302', 'fontSize': '20px', 'fontfamily': 'MiFuente', 'fontWeight': 'bold'}),
                html.Span("$62,888,163,585", style={'color': 'gray', 'fontSize': '20px', 'fontfamily': 'MiFuente'}),

                html.Span("\u00A0\u00A0\u00A0\u00A0\u00A0\u00A0", style={'fontSize': '20px'}),  # Espacios

                html.Span("Numero de posiciones (negociable): ", style={'color': '#D94302', 'fontSize': '20px', 'fontfamily': 'MiFuente', 'fontWeight': 'bold'}),
                html.Span("10", style={'color': 'gray', 'fontSize': '20px', 'fontfamily': 'MiFuente'}),

                html.Span("\u00A0\u00A0\u00A0\u00A0\u00A0\u00A0", style={'fontSize': '20px'}),  # Espacios

                html.Span("Patrimonio Tecnico: ", style={'color': '#D94302', 'fontSize': '20px', 'fontfamily': 'MiFuente', 'fontWeight': 'bold'}),
                html.Span("$500.000.000.000", style={'color': 'gray', 'fontSize': '20px', 'fontfamily': 'MiFuente'})
            ], style={'textAlign': 'center', 'marginTop': '50px'}),


            html.Br(), html.Br(),

            dash_table.DataTable(
                data=df.to_dict('records'),
                columns=[{"name": i, "id": i} for i in df.columns],
                style_table={'width': '70%', 'margin': 'auto'},
                style_header={'backgroundColor': '#FE4902', 'color': 'white', 'textAlign': 'center', 'fontFamily': 'MiFuente'},
                style_data={'backgroundColor': 'white', 'color': 'black', 'textAlign': 'center', 'border': '1px solid #ddd', 'fontFamily': 'MiFuente'}
            ),
            html.Br(),
            html.Br(),
            dcc.Graph(figure=fig_line, style={'width': '48%', 'display': 'inline-block'}),
            html.Div([
                dcc.Dropdown(id='drop1', options=[{'label': i, 'value': i} for i in ['Sector', 'País', 'Moneda', 'Tipo de Activo', 'Calificación Crediticia']], 
                             value='Moneda', style={'fontFamily': 'MiFuente'}),

                dcc.Graph(id='pie', figure=pie)
            ], style={'width': '48%', 'display': 'inline-block', 'vertical-align': 'top'}),

            html.Div([
    html.Div([
        dcc.Tabs(id="tabs2", value='tab-cupos', children=[
            dcc.Tab(label='Cupos', value='tab-cupos', style={"color": "#FE4902"}),
            dcc.Tab(label='Participaciones', value='tab-participacion', style={"color": "#FE4902"})
        ], style={'fontFamily': 'MiFuente'}),

        html.Br(), html.Br(),

        # Contenedor donde se actualizará la tabla
        html.Div(id="table-container"),

        # Botón para exportar
        html.Button("Exportar a Excel", id="export-button", style={"margin": "10px"}),

        # Componente de descarga
        dcc.Download(id="download-dataframe-xlsx")
    ], style={'flex': '1', 'display': 'flex', 'flexDirection': 'column'}),  # Ajustar tamaño del panel izquierdo

    html.Div([
        dcc.Graph(id='yiledcurve', figure=fig_curvas)
    ], style={'flex': '3', 'display': 'flex', 'justifyContent': 'flex-end'})  # Mover más a la derecha

], style={'display': 'flex', 'gap': '20px', 'alignItems': 'center'})



        ])
    else:
        return html.Div([

            html.Br(), html.Br(), html.Br(),

            html.Div(id='metricas-riesgo'),
            html.Div([
                html.Div([
                    dcc.Loading(id="loading-var_historico", type="circle", children=dcc.Graph(id='var_historico', figure={})),
                ], style={'width': '50%'}),
                html.Div([
                    dcc.Loading(id="loading-hist", type="circle", children=dcc.Graph(id='hist', figure={})),
                ], style={'width': '50%'})
            ], style={'display': 'flex', 'width': '100%'}),
            
            html.Div([
                html.Div([
                    dcc.Graph(figure=fig)
                ], style={'width': '50%'}),
                html.Div([
                    html.Br(), html.Br(),
                    html.H3("Seleccione horizonte de inversión (en semanas)", style={'textAlign': 'center', 'color': '#D94302','fontFamily': 'MiFuente'}),
                    dcc.Slider(id='slider', min=4, max=52, step=4, marks={i: str(i) for i in range(4, 53, 4)}, value=4),
                    html.H3("Seleccione nivel de confianza", style={'textAlign': 'center', 'color': '#D94302','fontFamily': 'MiFuente'}),
                    dcc.Slider(id='slider2', min=85, max=99, step=1, marks={i: str(i) for i in range(85, 100, 1)}, value=99)
                ], style={'width': '50%', 'paddingTop': '20px', 'textAlign': 'center'})
            ], style={'display': 'flex', 'width': '100%'})
        ])

@callback(Output('var_historico', 'figure'), Input('slider', 'value'), Input('slider2', 'value'))
def update_var_historico(semanas, confi):
    return data_VaRhist(sigma, dt, mu, semanas, confi)

@callback(Output('hist', 'figure'), Input('slider', 'value'), Input('slider2', 'value'))

def update_hist(semanas, confi):
    return histogram(s_0, sigma, dt, mu, semanas, confi)


# Callback para actualizar la tabla según el tab seleccionado
@callback(
    Output("table-container", "children"),
    Input("tabs2", "value")
)


def update_table(selected_tab):
    df_selected = df_cupos if selected_tab == "tab-cupos" else df_participacion
    titulo_principal = "Tabla de cupos" if selected_tab == "tab-cupos" else "Tabla de participaciones"
    titulo_secundario = "Cupo por emisor" if selected_tab == "tab-cupos" else "Participación por emisor"
    
    return html.Div([
        html.Div([
            html.Span(titulo_principal, style={'color': '#D94302', 'fontSize': '24px'}),
            html.Span(f": {titulo_secundario}", style={'color': 'gray', 'fontSize': '24px'})
        ], style={'textAlign': 'center', 'display': 'inline-block', 'white-space': 'nowrap', 'margin-top': '10px', 'margin-bottom': '20px'}),
        
        html.Div([
            dash_table.DataTable(
                data=df_selected.to_dict('records'),
                columns=[{"name": i, "id": i} for i in df_selected.columns],
                style_table={'width': '100%', 'max-width': '400px', 'margin': 'auto'},
                style_cell={'fontSize': '12px', 'padding': '5px'},
                style_header={'backgroundColor': '#FE4902', 'color': 'white', 'textAlign': 'center', 'fontFamily': 'MiFuente'},
                style_data={'backgroundColor': 'white', 'color': 'black', 'textAlign': 'center', 'border': '1px solid #ddd', 'fontFamily': 'MiFuente'}
            )
        ], style={'display': 'flex', 'justifyContent': 'center'})
    ])

# Callback para exportar la tabla actual a Excel
@callback(
    Output("download-dataframe-xlsx", "data"),
    Input("export-button", "n_clicks"),
    State("tabs2", "value"),
    prevent_initial_call=True
)


def export_to_excel(n_clicks, selected_tab):
    df_selected = df_cupos if selected_tab == "tab-cupos" else df_participacion
    buffer = io.BytesIO()
    df_selected.to_excel(buffer, index=False)
    buffer.seek(0)
    return dcc.send_bytes(buffer.getvalue(), "exported_data.xlsx")


@callback(
    Output('pie', 'figure'),
    Input('drop1', 'value')
)
def update_pie(selected_value):
    # Crear nuevo gráfico de pie basado en la selección
    fig_pie = px.pie(
        df_pesos, 
        names=selected_value, 
        values="Peso", 
        hole=0.4, 
        color_discrete_sequence=naranja_degradado
    )

    fig_pie.update_layout(
        title={
                'text': f"Composición por {selected_value}",
                 'x': 0.5,  # Centrar el título
                 'xanchor': 'center',
                 'yanchor': 'top'
                            },

                    title_font=dict(
                        family="MiFuente",  # Tipo de letra
                        size=24,         # Tamaño del título
                        color="#D94302"      # Color del título
                    )
    )
    return fig_pie

@app.callback(
    Output('metricas-riesgo', 'children'),
    Input('slider', 'value'),
    Input('slider2', 'value')
)


def update_text(horizonte, confianza):

    np.random.seed(45)
    percentil = 100 - confianza

    df_sim = sim_caminos(s_0, sigma, dt, mu, horizonte)

    distr_retornos = (df_sim.iloc[:, -1] - s_0) / s_0

    VaR = np.percentile(distr_retornos, percentil)

    VaR = f"{VaR* 100:.2f}%"

    return html.Div([
        html.Span(f"VaR : {VaR}", style={'color': '#D94302', 'fontSize': '20px', 'fontFamily': 'MiFuente', 'fontWeight': 'bold'}),
        html.Span("\u00A0\u00A0\u00A0\u00A0\u00A0\u00A0", style={'fontSize': '20px'}),  # Espacios
        html.Span(f"Sharpe ratio : 15%", style={'color': '#D94302', 'fontSize': '20px', 'fontFamily': 'MiFuente', 'fontWeight': 'bold'})
    ], style={'textAlign': 'center'})
    

if __name__ == '__main__':
    app.run_server(debug=True, jupyter_mode="external", port=8060)
