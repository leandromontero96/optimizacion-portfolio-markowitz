import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import json

st.set_page_config(
    page_title="Portfolio Optimizer",
    page_icon="💼",
    layout="wide"
)

@st.cache_data
def load_data():
    prices = pd.read_csv('../data/historical_prices.csv', index_col=0, parse_dates=True)
    returns = pd.read_csv('../data/returns.csv', index_col=0, parse_dates=True)
    with open('../data/optimization_results.json', 'r') as f:
        results = json.load(f)[0]
    return prices, returns, results

def main():
    st.title("💼 Optimización de Portfolio - Teoría Moderna (Markowitz)")
    st.markdown("### Análisis y Optimización de Inversiones")

    prices, returns, results = load_data()

    st.sidebar.title("Navegación")
    page = st.sidebar.radio(
        "Seleccione página:",
        ["📊 Overview", "🎯 Portfolio Óptimo", "📈 Performance", "⚠️ Análisis de Riesgo"]
    )

    if page == "📊 Overview":
        show_overview(prices, returns, results)
    elif page == "🎯 Portfolio Óptimo":
        show_optimal_portfolio(results, returns)
    elif page == "📈 Performance":
        show_performance(prices, returns, results)
    elif page == "⚠️ Análisis de Riesgo":
        show_risk_analysis(returns, results)

def show_overview(prices, returns, results):
    st.header("📊 Overview del Portfolio")

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("Retorno Anualizado", f"{results['backtest_metrics']['annual_return']:.2f}%")

    with col2:
        st.metric("Volatilidad", f"{results['backtest_metrics']['annual_vol']:.2f}%")

    with col3:
        st.metric("Sharpe Ratio", f"{results['backtest_metrics']['sharpe_ratio']:.4f}")

    with col4:
        st.metric("Max Drawdown", f"{results['backtest_metrics']['Max_Drawdown']:.2f}%")

    st.markdown("---")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Precios Históricos")
        fig = go.Figure()
        for col in prices.columns:
            fig.add_trace(go.Scatter(
                x=prices.index,
                y=prices[col],
                name=col,
                mode='lines'
            ))
        fig.update_layout(
            height=400,
            xaxis_title="Fecha",
            yaxis_title="Precio ($)",
            hovermode='x unified'
        )
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.subheader("Retornos Anualizados")
        annual_returns = returns.mean() * 252 * 100
        fig = px.bar(
            x=annual_returns.index,
            y=annual_returns.values,
            labels={'x': 'Activo', 'y': 'Retorno (%)'},
            color=annual_returns.values,
            color_continuous_scale='RdYlGn'
        )
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)

    st.subheader("Matriz de Correlación")
    corr_matrix = returns.corr()
    fig = px.imshow(
        corr_matrix,
        text_auto='.2f',
        aspect='auto',
        color_continuous_scale='RdBu',
        zmin=-1,
        zmax=1
    )
    fig.update_layout(height=500)
    st.plotly_chart(fig, use_container_width=True)

def show_optimal_portfolio(results, returns):
    st.header("🎯 Portfolio Óptimo")

    tab1, tab2 = st.tabs(["Máximo Sharpe Ratio", "Mínima Varianza"])

    with tab1:
        st.subheader("Portfolio de Máximo Sharpe Ratio")

        weights_sharpe = results['optimal_weights_sharpe']
        weights_df = pd.DataFrame(list(weights_sharpe.items()), columns=['Activo', 'Peso'])
        weights_df = weights_df[weights_df['Peso'] > 0.01].sort_values('Peso', ascending=False)

        col1, col2 = st.columns([1, 1])

        with col1:
            st.markdown("#### Distribución de Activos")
            fig = px.pie(
                weights_df,
                values='Peso',
                names='Activo',
                hole=0.4,
                title="Asset Allocation"
            )
            fig.update_traces(textposition='inside', textinfo='percent+label')
            st.plotly_chart(fig, use_container_width=True)

        with col2:
            st.markdown("#### Pesos por Activo")
            fig = px.bar(
                weights_df,
                x='Activo',
                y='Peso',
                title="Pesos del Portfolio"
            )
            fig.update_traces(text=weights_df['Peso'].apply(lambda x: f'{x*100:.1f}%'))
            fig.update_layout(height=350)
            st.plotly_chart(fig, use_container_width=True)

        st.markdown("#### Tabla de Asignación")
        weights_df['Peso (%)'] = weights_df['Peso'] * 100
        st.dataframe(weights_df[['Activo', 'Peso (%)']].style.format({'Peso (%)': '{:.2f}'}), use_container_width=True)

    with tab2:
        st.subheader("Portfolio de Mínima Varianza")

        weights_minvar = results['optimal_weights_minvar']
        weights_df = pd.DataFrame(list(weights_minvar.items()), columns=['Activo', 'Peso'])
        weights_df = weights_df[weights_df['Peso'] > 0.01].sort_values('Peso', ascending=False)

        col1, col2 = st.columns([1, 1])

        with col1:
            fig = px.pie(
                weights_df,
                values='Peso',
                names='Activo',
                hole=0.4,
                title="Asset Allocation (Min Variance)"
            )
            fig.update_traces(textposition='inside', textinfo='percent+label')
            st.plotly_chart(fig, use_container_width=True)

        with col2:
            fig = px.bar(
                weights_df,
                x='Activo',
                y='Peso',
                title="Pesos del Portfolio"
            )
            fig.update_traces(text=weights_df['Peso'].apply(lambda x: f'{x*100:.1f}%'))
            fig.update_layout(height=350)
            st.plotly_chart(fig, use_container_width=True)

        weights_df['Peso (%)'] = weights_df['Peso'] * 100
        st.dataframe(weights_df[['Activo', 'Peso (%)']].style.format({'Peso (%)': '{:.2f}'}), use_container_width=True)

def show_performance(prices, returns, results):
    st.header("📈 Performance del Portfolio")

    weights_sharpe = np.array(list(results['optimal_weights_sharpe'].values()))
    portfolio_returns = (returns @ weights_sharpe)
    portfolio_value = 100000 * (1 + portfolio_returns).cumprod()

    col1, col2, col3 = st.columns(3)

    with col1:
        total_return = (portfolio_value.iloc[-1] / 100000 - 1) * 100
        st.metric("Retorno Total", f"{total_return:.2f}%")

    with col2:
        final_value = portfolio_value.iloc[-1]
        st.metric("Valor Final", f"${final_value:,.2f}")

    with col3:
        sharpe = results['backtest_metrics']['sharpe_ratio']
        st.metric("Sharpe Ratio", f"{sharpe:.4f}")

    st.subheader("Evolución del Portfolio")
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=portfolio_value.index,
        y=portfolio_value.values,
        fill='tozeroy',
        name='Portfolio Value',
        line=dict(color='#2ecc71', width=2)
    ))
    fig.update_layout(
        height=400,
        xaxis_title="Fecha",
        yaxis_title="Valor del Portfolio ($)",
        hovermode='x unified'
    )
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Distribución de Retornos Diarios")
    col1, col2 = st.columns([2, 1])

    with col1:
        fig = px.histogram(
            portfolio_returns * 100,
            nbins=50,
            title="Distribución de Retornos Diarios (%)"
        )
        fig.add_vline(
            x=portfolio_returns.mean() * 100,
            line_dash="dash",
            line_color="red",
            annotation_text=f"Media: {portfolio_returns.mean()*100:.2f}%"
        )
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.markdown("#### Estadísticas")
        st.metric("Media", f"{portfolio_returns.mean()*100:.3f}%")
        st.metric("Mediana", f"{portfolio_returns.median()*100:.3f}%")
        st.metric("Desv. Std", f"{portfolio_returns.std()*100:.3f}%")
        st.metric("Skewness", f"{portfolio_returns.skew():.3f}")
        st.metric("Kurtosis", f"{portfolio_returns.kurtosis():.3f}")

def show_risk_analysis(returns, results):
    st.header("⚠️ Análisis de Riesgo")

    weights_sharpe = np.array(list(results['optimal_weights_sharpe'].values()))
    portfolio_returns = (returns @ weights_sharpe)

    col1, col2, col3 = st.columns(3)

    with col1:
        var_95 = results['backtest_metrics']['VaR_95']
        st.metric("VaR (95%)", f"{var_95:.2f}%")

    with col2:
        cvar_95 = results['backtest_metrics']['CVaR_95']
        st.metric("CVaR (95%)", f"{cvar_95:.2f}%")

    with col3:
        max_dd = results['backtest_metrics']['Max_Drawdown']
        st.metric("Max Drawdown", f"{max_dd:.2f}%")

    st.markdown("---")

    st.subheader("Drawdown del Portfolio")
    cumulative_returns = (1 + portfolio_returns).cumprod()
    running_max = cumulative_returns.expanding().max()
    drawdown = (cumulative_returns - running_max) / running_max * 100

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=drawdown.index,
        y=drawdown.values,
        fill='tozeroy',
        name='Drawdown',
        line=dict(color='red', width=1),
        fillcolor='rgba(255,0,0,0.3)'
    ))
    fig.update_layout(
        height=400,
        xaxis_title="Fecha",
        yaxis_title="Drawdown (%)",
        hovermode='x unified'
    )
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Análisis de Volatilidad Rolling")
    rolling_vol = portfolio_returns.rolling(window=30).std() * np.sqrt(252) * 100

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=rolling_vol.index,
        y=rolling_vol.values,
        name='Volatilidad Rolling (30d)',
        line=dict(color='orange', width=2)
    ))
    fig.add_hline(
        y=results['backtest_metrics']['annual_vol'],
        line_dash="dash",
        line_color="red",
        annotation_text=f"Volatilidad Anual: {results['backtest_metrics']['annual_vol']:.2f}%"
    )
    fig.update_layout(
        height=400,
        xaxis_title="Fecha",
        yaxis_title="Volatilidad (%)",
        hovermode='x unified'
    )
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Interpretación de Métricas de Riesgo")

    with st.expander("📖 ¿Qué es VaR (Value at Risk)?"):
        st.markdown(f"""
        **VaR al 95%: {var_95:.2f}%**

        El VaR (Value at Risk) indica la pérdida máxima esperada con un 95% de confianza en un día.
        En este caso, hay un 5% de probabilidad de perder más de {abs(var_95):.2f}% en un día.
        """)

    with st.expander("📖 ¿Qué es CVaR (Conditional VaR)?"):
        st.markdown(f"""
        **CVaR al 95%: {cvar_95:.2f}%**

        El CVaR (Expected Shortfall) es la pérdida promedio cuando se excede el VaR.
        Indica cuánto se puede perder en los peores casos (5% más extremo).
        """)

    with st.expander("📖 ¿Qué es el Max Drawdown?"):
        st.markdown(f"""
        **Max Drawdown: {max_dd:.2f}%**

        Es la mayor caída desde un máximo histórico hasta el punto más bajo.
        Indica la peor pérdida que habría experimentado un inversor desde un pico.
        """)

if __name__ == "__main__":
    main()
