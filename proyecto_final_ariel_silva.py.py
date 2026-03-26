# -*- coding: utf-8 -*-
"""
Proyecto Final - Diplomado Ciencia de Datos para las Finanzas FEN

Alumno: Ariel Silva Echeverría
RUN: 19.743.239-K

El presente proyecto tiene como objetivo desarrollar una herramienta de apoyo
a la toma de decisiones de inversión mediante el uso de técnicas de Machine 
Learning y análisis financiero (análisis fundamental).

Se implementa un modelo basado en redes neuronales recurrentes (LSTM) para
 modelar y predecir el comportamiento de los retornos de activos financieros, 
 capturando patrones temporales en series de tiempo (inversión a corto plazo).

Adicionalmente, se incorpora un módulo de análisis fundamental que permite 
estimar el valor intrínseco de las acciones, junto con métricas de riesgo como 
el Beta,generando una evaluación integral del activo.

Finalmente, se desarrolla una aplicación interactiva utilizando Streamlit,
la cual permite al usuario ingresar cualquier ticker del mercado y obtener
un análisis automatizado que incluye:

- Precio actual
- Valoración fundamental
- Proyección a largo plazo
- Riesgo (Beta)
- Recomendación de inversión

Este sistema busca integrar análisis cuantitativo y financiero en una sola
plataforma, facilitando la interpretación de datos complejos para la toma
de decisiones informadas.

Cabe destacar que el modelo constituye una herramienta de apoyo y no reemplaza
el juicio profesional en materia de inversión. Por lo tanto el modelo no es en
ningun caso una recomendación de inversión sino una herramienta de apoyo a la 
toma de decisiones.
"""
#%%
# Importación de librería
import numpy as np
import pandas as pd
import yfinance as yf

from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
#%%
# Descarga de datos reales de mis acciones de mi portafolio desde Yahoo Finance

tickers = ["GOOG", "MSFT", "MELI", "AAPL", "AMZN", "TSLA", "PEP", "NKE"]

data = yf.download(tickers, start="2015-01-01", end="2024-12-31")

# Extraemos correctamente precios ajustados
data = data["Close"]

print("Datos descargados:")
print(data.tail())
#%%
# Cálculo de retornos logarítmicos

returns = np.log(data / data.shift(1)).dropna()

print("Retornos calculados:")
print(returns.tail())
#%%
# División temporal para evitar data leakage
split_returns = int(0.8 * len(returns))

# Normalización con datos de entrenamiento
scaler = MinMaxScaler()
scaler.fit(returns[:split_returns])

scaled_data = scaler.transform(returns)

print("Datos normalizados:")
print(scaled_data[:5])
#%%
# Creación de secuencias para modelo RNN

window_size = 10

X = []
y = []

for i in range(window_size, len(scaled_data)):
    X.append(scaled_data[i-window_size:i])
    y.append(scaled_data[i])

X = np.array(X)
y = np.array(y)

print("Shape de X:", X.shape)
print("Shape de y:", y.shape)
#%%
# División entrenamiento 80% / prueba 20% (temporal)
split = int(0.8 * len(X))

X_train = X[:split]
X_test = X[split:]

y_train = y[:split]
y_test = y[split:]

print("Train shape:", X_train.shape)
print("Test shape:", X_test.shape)
#%%
# Construcción del modelo RNN que aprenderá patrones del mercado

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout

model = Sequential()

# Primera capa LSTM (memoria de largo plazo)
model.add(LSTM(
    units=64,
    return_sequences=True,
    input_shape=(X_train.shape[1], X_train.shape[2])
))

# Regularización
model.add(Dropout(0.2))

# Segunda capa LSTM
model.add(LSTM(units=32))

# Capa de salida
model.add(Dense(X_train.shape[2]))

# Compilación
model.compile(
    optimizer='adam',
    loss='huber'
)
model.summary()
#%%
# Entrenamiento del modelo

from tensorflow.keras.callbacks import EarlyStopping

early_stop = EarlyStopping(
    monitor='val_loss',
    patience=5,
    restore_best_weights=True
)

history = model.fit(
    X_train,
    y_train,
    epochs=50,
    batch_size=32,
    validation_data=(X_test, y_test),
    callbacks=[early_stop],
    verbose=1
)
#%%
# Predicciones

predictions = model.predict(X_test)

# Desnormalización
predictions_rescaled = scaler.inverse_transform(predictions)
y_test_rescaled = scaler.inverse_transform(y_test)
#%%
# Selección de un activo (ej: GOOG = índice 0)

activo = 0

y_real = y_test_rescaled[:, activo]
y_pred = predictions_rescaled[:, activo]
#%% 
# Evaluación estrategia (ejemplo con activo seleccionado)

senal = np.sign(y_pred)
estrategia = senal * y_real

retorno_estrategia = np.mean(estrategia)
retorno_anual_estrategia = (1 + retorno_estrategia)**252 - 1
vol_estrategia = np.std(estrategia)
# tasa libre de riesgo (aprox 3%)
rf = 0.03
sharpe_estrategia = (retorno_anual_estrategia - rf) / (vol_estrategia * np.sqrt(252))

print("Sharpe estrategia:", sharpe_estrategia)
#%%
# Evaluación MSE 

from sklearn.metrics import mean_squared_error

mse = mean_squared_error(y_real, y_pred)

print("MSE del modelo:", mse)
#%%
# Retornos reales (benchmark del modelo)
retornos_reales = y_real

retorno = np.mean(retornos_reales)
retorno_anual = (1 + retorno)**252 - 1

volatilidad = np.std(retornos_reales)

sharpe = (retorno_anual - rf) / (volatilidad * np.sqrt(252))

print("Retorno esperado anual:", retorno_anual)
print("Volatilidad:", volatilidad)
print("Sharpe Ratio:", sharpe)
#%%
# Benchmark de mercado (SPY) 

spy = yf.download("SPY", start="2015-01-01", end="2024-12-31")["Close"]

# retornos simples 
spy_returns = spy.pct_change().dropna()

# métricas
retorno_spy = spy_returns.mean()
retorno_spy_anual = retorno_spy * 252

vol_spy = spy_returns.std()

# Sharpe ratio 
sharpe_spy = (retorno_spy_anual - rf) / (vol_spy * np.sqrt(252))

# asegurar tipo float 
sharpe_spy = float(sharpe_spy)

print("Sharpe SPY:", sharpe_spy)
#%% 
# Análisis por activo

resultados = []

for i, ticker in enumerate(tickers):
    
    y_real = y_test_rescaled[:, i]
    y_pred = predictions_rescaled[:, i]
    
    #  Estrategia basada en el modelo
    senal = np.sign(y_pred)
    retornos_modelo = senal * y_real
    
    retorno = np.mean(retornos_modelo)
    retorno_anual = (1 + retorno)**252 - 1
    
    volatilidad = np.std(retornos_modelo)
    
    sharpe = (retorno_anual - rf) / (volatilidad * np.sqrt(252))
    
    resultados.append({
        "Ticker": ticker,
        "Retorno anual": retorno_anual,
        "Volatilidad": volatilidad,
        "Sharpe": sharpe
    })

df_resultados = pd.DataFrame(resultados)
#%%
# Estrategia: solo invertimos cuando modelo predice positivo
estrategia = senal * y_real

# Métricas estrategia
retorno_estrategia = np.mean(estrategia)
retorno_anual_estrategia = (1 + retorno_estrategia)**252 - 1
vol_estrategia = np.std(estrategia)

sharpe_estrategia = (retorno_anual_estrategia - rf) / (vol_estrategia * np.sqrt(252))
# Ordenar por Sharpe 
df_resultados = df_resultados.sort_values(by="Sharpe", ascending=False)

print(df_resultados)
#%%
#  lógica de recomendación (alineada con mercado)

def recomendacion_final(sharpe_activo, sharpe_mercado, retorno_predicho):

    # Calidad relativa al mercado
    if sharpe_activo > sharpe_mercado:
        calidad = "ALTA"
    elif sharpe_activo > 0:
        calidad = "MEDIA"
    else:
        calidad = "BAJA"

    # Señal del modelo (timing)
    if retorno_predicho > 0:
        timing = "ALCISTA"
    else:
        timing = "BAJISTA"

    # Decisión final
    if calidad == "ALTA" and timing == "ALCISTA":
        decision = "COMPRAR"
    elif calidad == "MEDIA" and timing == "ALCISTA":
        decision = "MANTENER"
    elif calidad == "ALTA" and timing == "BAJISTA":
        decision = "MANTENER (corrección esperada)"
    else:
        decision = "VENDER"

    return decision

print(df_resultados)
#%% 
# Aplicar recomendación correctamente
recomendaciones = []

for i, row in df_resultados.iterrows():
    
    ticker = row["Ticker"]
    sharpe_activo = row["Sharpe"]
    
    # reconstruimos retorno del modelo por activo
    y_pred_activo = predictions_rescaled[:, tickers.index(ticker)]
    retorno_ml = np.mean(y_pred_activo)
    
    decision = recomendacion_final(sharpe_activo, sharpe_spy, retorno_ml)
    
    recomendaciones.append(decision)

df_resultados["Recomendación"] = recomendaciones

print(df_resultados)
#%%
# Conclusiones automáticas

mejor_activo = df_resultados.iloc[0]

print("\n--- CONCLUSIONES ---")

print(f"El activo con mejor desempeño ajustado por riesgo es {mejor_activo['Ticker']},")
print(f"con un Sharpe Ratio de {round(mejor_activo['Sharpe'],2)}.")

print("El modelo logra capturar parcialmente la dirección de los retornos,")
print("aunque presenta limitaciones para modelar eventos de alta volatilidad.")

print("Se concluye que el modelo es útil como herramienta de apoyo")
print("para decisiones de inversión, especialmente en la identificación de tendencias,")
print("pero no debe ser utilizado como único criterio de decisión.")

print("Una mejora futura sería incorporar factores de riesgo como Fama-French")
print("y variables macroeconómicas para aumentar la robustez del modelo.")
#%%
# FUNCIÓN FINAL TIPO HERRAMIENTA
def analizar_activo(ticker, data, predictions_rescaled, y_test_rescaled, tickers, sharpe_spy, rf=0.03):

    print("DEBUG: entrando a función")

    # VALIDACIÓN ML
    usar_ml = ticker in tickers

    if not usar_ml:
        print(f"⚠️ {ticker} no está en el modelo ML → usando solo fundamental")

    # PRECIO ACTUAL

    try:
     precio_actual = yf.Ticker(ticker).history(period="1d")["Close"].iloc[-1]
    except:
     precio_actual = data[ticker].iloc[-1]  # fallback

    # DATOS ML 
    
    if usar_ml:
        idx = tickers.index(ticker)
        y_real = y_test_rescaled[:, idx]
        y_pred = predictions_rescaled[:, idx]

        retorno_ml = float(np.mean(y_pred))
        volatilidad = np.std(y_real) * np.sqrt(252)

        retorno_real = np.mean(y_real)
        retorno_real_anual = (1 + retorno_real)**252 - 1

        sharpe = float((retorno_real_anual - rf) / (volatilidad * np.sqrt(252)))

    else:
        retorno_ml = 0
        volatilidad = 0
        sharpe = 0

    # BETA 
  
    spy = yf.download("SPY", period="1y")["Close"]
    spy.index = spy.index.tz_localize(None)
    
    spy_returns = spy.pct_change().dropna()
    
    activo_prices = yf.download(ticker, period="1y")["Close"]
    activo_prices.index = activo_prices.index.tz_localize(None)
    
    activo_returns = activo_prices.pct_change().dropna()
    
    df_beta = pd.concat([activo_returns, spy_returns], axis=1).dropna()
    df_beta.columns = ["activo", "spy"]
    
    cov = np.cov(df_beta["activo"], df_beta["spy"])[0,1]
    var = np.var(df_beta["spy"])
    
    beta = cov / var

    #  Analisis Fundamental
    
    try:
        info = yf.Ticker(ticker).info
    except:
        info = {}

    pe = info.get("trailingPE", None)
    growth = info.get("earningsQuarterlyGrowth", None)
    
    if growth is None:
     growth = 0.05
    
    # Convertir a crecimiento anual aproximado
    growth = growth * 4
    
    # Capar crecimiento 
    growth = min(max(growth, 0.02), 0.15)

    discount_rate = 0.10
    n = 5

    valor_intrinseco = precio_actual * ((1 + growth) / (1 + discount_rate)) ** n
    valor_intrinseco = valor_intrinseco * 0.8
    
    # PROYECCIÓN
    
    precio_2y = precio_actual * (1 + growth)**2
    precio_5y = precio_actual * (1 + growth)**5
    precio_10y = precio_actual * (1 + growth)**10

    retorno_2y = (precio_2y / precio_actual - 1) * 100
    retorno_5y = (precio_5y / precio_actual - 1) * 100
    retorno_10y = (precio_10y / precio_actual - 1) * 100

    # VALORACIÓN
   
    if valor_intrinseco > precio_actual * 1.05:
        estado = "Subvalorado"
    elif valor_intrinseco < precio_actual * 0.95:
        estado = "Sobrevalorado"
    else:
        estado = "Valor justo"
   
    # DECISIÓN 

    if estado == "Subvalorado" and beta < 1.2:
        decision = "🟢 COMPRAR"
    elif estado == "Valor justo":
        decision = "🟡 MANTENER"
    else:
        decision = "🔴 VENDER"

    # OUTPUT
    
    print("\n" + "="*35)
    print("📊 ANÁLISIS DE INVERSIÓN")
    print("="*35)

    print(f"\nTicker: {ticker}")
    print(f"Precio actual: {round(precio_actual,2)}")

    print("\nVALORACIÓN FUNDAMENTAL:")
    print(f"PE ratio: {round(pe,2) if pe else 'N/A'}")
    print(f"Valor intrínseco: {round(valor_intrinseco,2)}")
    print(f"Estado: {estado}")

    print("\nLARGO PLAZO:")
    print(f"2 años → {round(precio_2y,2)} ({round(retorno_2y,1)}%)")
    print(f"5 años → {round(precio_5y,2)} ({round(retorno_5y,1)}%)")
    print(f"10 años → {round(precio_10y,2)} ({round(retorno_10y,1)}%)")

    if usar_ml:
        precio_esperado = precio_actual * np.exp(retorno_ml * 30)

        print("\nCORTO PLAZO (ML):")
        print(f"Precio esperado: {round(precio_esperado,2)}")
        print(f"Retorno esperado: {round(retorno_ml*100,2)}%")

    print("\nRIESGO:")
    print(f"Volatilidad: {round(volatilidad*100,2)}%")
    print(f"Sharpe: {round(sharpe,2)}")
    print(f"Beta: {round(beta,2)}")

    print("\n📌 RECOMENDACIÓN FINAL:")
    print(f"→ {decision}")
    print("="*35)
    print("\n⚠️ Nota: Este modelo es una herramienta de apoyo y no reemplaza análisis financiero profesional.")
 #%%
analizar_activo("TSLA", data, predictions_rescaled, y_test_rescaled, tickers, sharpe_spy)

