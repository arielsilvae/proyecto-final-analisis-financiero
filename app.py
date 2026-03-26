import streamlit as st
import yfinance as yf
import numpy as np
import pandas as pd

st.set_page_config(page_title="Calculadora de Valor Intrínseco", layout="centered")

st.title("📊 Calculadora de Valor Intrínseco")
st.write("Análisis fundamental + proyección financiera")

# -----------------------------
# INPUT
# -----------------------------
ticker = st.text_input("Ingresa un ticker (ej: AAPL, TSLA, MSFT)", value="AAPL")

# -----------------------------
# FUNCIÓN PRINCIPAL
# -----------------------------
def analizar(ticker):

    # -----------------------------
    # PRECIO ACTUAL
    # -----------------------------
    data = yf.Ticker(ticker)
    hist = data.history(period="1y")

    if hist.empty:
        st.error("Ticker no válido o sin datos")
        return

    precio_actual = hist["Close"].iloc[-1]

    # -----------------------------
    # FUNDAMENTAL
    # -----------------------------
    try:
        info = data.info
    except:
        info = {}

    pe = info.get("trailingPE", None)
    growth = info.get("earningsQuarterlyGrowth", None)

    if growth is None:
        growth = 0.05

    growth = growth * 4
    growth = min(max(growth, 0.02), 0.15)

    discount_rate = 0.10
    n = 5

    valor_intrinseco = precio_actual * ((1 + growth) / (1 + discount_rate)) ** n
    valor_intrinseco = valor_intrinseco * 0.8

    # -----------------------------
    # PROYECCIÓN
    # -----------------------------
    precio_2y = precio_actual * (1 + growth)**2
    precio_5y = precio_actual * (1 + growth)**5
    precio_10y = precio_actual * (1 + growth)**10

    retorno_2y = (precio_2y / precio_actual - 1) * 100
    retorno_5y = (precio_5y / precio_actual - 1) * 100
    retorno_10y = (precio_10y / precio_actual - 1) * 100

    # -----------------------------
    # BETA (CORREGIDO)
    # -----------------------------
    spy = yf.download("SPY", period="1y")["Close"]
    spy.index = spy.index.tz_localize(None)

    returns_spy = spy.pct_change().dropna()

    activo = yf.download(ticker, period="1y")["Close"]
    activo.index = activo.index.tz_localize(None)

    returns_activo = activo.pct_change().dropna()

    df_beta = pd.concat([returns_activo, returns_spy], axis=1).dropna()
    df_beta.columns = ["activo", "spy"]

    cov = np.cov(df_beta["activo"], df_beta["spy"])[0,1]
    var = np.var(df_beta["spy"])

    beta = cov / var

    # -----------------------------
    # VALORACIÓN
    # -----------------------------
    if valor_intrinseco > precio_actual * 1.1:
        estado = "🟢 Subvalorado"
    elif valor_intrinseco < precio_actual * 0.9:
        estado = "🔴 Sobrevalorado"
    else:
        estado = "🟡 Valor justo"

    # -----------------------------
    # DECISIÓN FINAL
    # -----------------------------
    if estado == "🟢 Subvalorado" and beta < 1.2:
        decision = "🟢 COMPRAR"
    elif estado == "🟡 Valor justo":
        decision = "🟡 MANTENER"
    else:
        decision = "🔴 VENDER"

    # -----------------------------
    # OUTPUT
    # -----------------------------
    st.subheader("Resultado")

    st.write(f"**Ticker:** {ticker}")
    st.write(f"**Precio actual:** {round(precio_actual,2)} USD")

    st.markdown("### 📊 Valoración Fundamental")
    st.write(f"PE ratio: {round(pe,2) if pe else 'N/A'}")
    st.write(f"Valor intrínseco: {round(valor_intrinseco,2)}")
    st.write(f"Estado: {estado}")

    st.markdown("### 📈 Proyección")
    st.write(f"2 años → {round(precio_2y,2)} ({round(retorno_2y,1)}%)")
    st.write(f"5 años → {round(precio_5y,2)} ({round(retorno_5y,1)}%)")
    st.write(f"10 años → {round(precio_10y,2)} ({round(retorno_10y,1)}%)")

    st.markdown("### ⚠️ Riesgo")
    st.write(f"Beta: {round(beta,2)}")

    st.markdown("### 📌 Recomendación Final")
    st.success(decision)

    st.markdown("---")
    st.info("Este modelo es una herramienta de apoyo y no reemplaza análisis financiero profesional.")

# -----------------------------
# BOTÓN
# -----------------------------
if st.button("Analizar"):
    analizar(ticker)