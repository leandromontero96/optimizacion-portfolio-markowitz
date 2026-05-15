# 💼 Optimización de Portfolio - Teoría Moderna de Markowitz

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue)](https://www.python.org/)
[![scipy](https://img.shields.io/badge/scipy-1.11%2B-yellow)](https://scipy.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.28%2B-FF4B4B)](https://streamlit.io/)

## 📋 Descripción del Proyecto

Sistema completo de **Optimización de Portfolio** basado en la Teoría Moderna de Portfolio (MPT) de Harry Markowitz. Implementa optimización cuadrática, cálculo de frontera eficiente, backtesting avanzado, y métricas de riesgo (VaR, CVaR, Max Drawdown).

**Resultados del Portfolio Óptimo**:
- **Retorno Anualizado**: 20.38%
- **Volatilidad**: 10.79%
- **Sharpe Ratio**: 1.61
- **Max Drawdown**: -10.26%

---

## 🎯 Características Principales

### 1. Optimización de Portfolio
- ✅ **Máximo Sharpe Ratio** (mejor ratio riesgo-retorno)
- ✅ **Mínima Varianza** (menor riesgo posible)
- ✅ **Frontera Eficiente** (5,000 portfolios simulados)
- ✅ **Constraints flexibles** (sin short-selling, límites por activo)

### 2. Métricas de Riesgo
- 📊 **VaR (Value at Risk)** al 95% de confianza
- 📉 **CVaR (Conditional VaR)** / Expected Shortfall
- 📈 **Maximum Drawdown**
- 📊 **Volatilidad rolling**
- 📊 **Correlaciones entre activos**

### 3. Backtesting
- ⏱️ **5 años de datos históricos** (2020-2024)
- 💰 **Performance tracking**
- 📊 **Análisis de drawdown**
- 📈 **Distribución de retornos**

### 4. Dashboard Interactivo
- 🎨 **Visualizaciones con Plotly**
- 📊 **Asset allocation interactiva**
- 📈 **Gráficos de performance**
- ⚠️ **Análisis de riesgo detallado**

---

## 📊 Resultados de Optimización

### Portfolio Óptimo (Máximo Sharpe Ratio)

| Activo | Peso | Descripción |
|--------|------|-------------|
| **SPY** | 41.09% | S&P 500 ETF |
| **QQQ** | 27.21% | Nasdaq-100 ETF |
| **GLD** | 31.70% | Gold ETF |

**Métricas del Portfolio**:
- Retorno Esperado: **19.14%**
- Volatilidad: **10.79%**
- Sharpe Ratio: **1.50**

### Backtesting (5 años)

| Métrica | Valor |
|---------|-------|
| Capital Inicial | $100,000 |
| Capital Final | $261,117 |
| Retorno Total | **161.12%** |
| Retorno Anualizado | **20.38%** |
| Volatilidad Anualizada | **10.79%** |
| Sharpe Ratio | **1.61** |

### Métricas de Riesgo

| Métrica | Valor | Interpretación |
|---------|-------|----------------|
| **VaR (95%)** | -1.01% | Pérdida máxima esperada (95% confianza) en 1 día |
| **CVaR (95%)** | -1.24% | Pérdida promedio en el 5% peor de casos |
| **Max Drawdown** | -10.26% | Mayor caída desde un máximo histórico |

---

## 🏗️ Arquitectura del Proyecto

```
optimizacion-portfolio/
│
├── data/
│   ├── historical_prices.csv       # Precios históricos (5 años)
│   ├── returns.csv                 # Retornos diarios
│   ├── optimization_results.json   # Resultados de optimización
│   └── portfolio_analysis.png      # Gráficos de análisis
│
├── src/
│   ├── portfolio_optimizer.py      # Motor de optimización
│   └── app_dashboard.py            # Dashboard Streamlit
│
├── notebooks/
│   └── exploratory_analysis.ipynb  # Análisis exploratorio
│
└── README.md
```

---

## 🚀 Instalación y Uso

### 1. Clonar Repositorio

```bash
git clone https://github.com/tu-usuario/optimizacion-portfolio.git
cd optimizacion-portfolio
```

### 2. Instalar Dependencias

```bash
pip install pandas numpy scipy matplotlib seaborn streamlit plotly joblib
```

### 3. Ejecutar Optimización

```bash
cd src
python portfolio_optimizer.py
```

### 4. Visualizar Dashboard

```bash
streamlit run app_dashboard.py
```

Dashboard disponible en: `http://localhost:8501`

---

## 📐 Metodología

### 1. Teoría Moderna de Portfolio (Markowitz)

El objetivo es encontrar la combinación óptima de activos que maximice el ratio de Sharpe:

$$
\text{Sharpe Ratio} = \frac{R_p - R_f}{\sigma_p}
$$

Donde:
- $R_p$ = Retorno esperado del portfolio
- $R_f$ = Tasa libre de riesgo (3%)
- $\sigma_p$ = Volatilidad (desviación estándar) del portfolio

### 2. Optimización Cuadrática

**Función objetivo** (maximizar Sharpe):
$$
\max_w \frac{w^T \mu - r_f}{\sqrt{w^T \Sigma w}}
$$

**Constraints**:
- $\sum_{i=1}^{n} w_i = 1$ (suma de pesos = 100%)
- $0 \leq w_i \leq 1$ (sin short-selling)

Donde:
- $w$ = vector de pesos
- $\mu$ = vector de retornos esperados
- $\Sigma$ = matriz de covarianza

### 3. Frontera Eficiente

La frontera eficiente representa todos los portfolios que maximizan retorno para un nivel de riesgo dado.

Se calcula mediante:
1. Simulación de 5,000 portfolios aleatorios
2. Cálculo de retorno y volatilidad para cada uno
3. Identificación de portfolios óptimos

### 4. Métricas de Riesgo

**VaR (Value at Risk)**:
$$
\text{VaR}_\alpha = \text{Percentil}_\alpha(\text{Retornos})
$$

**CVaR (Conditional VaR)**:
$$
\text{CVaR}_\alpha = E[R | R \leq \text{VaR}_\alpha]
$$

**Max Drawdown**:
$$
\text{DD}(t) = \frac{\text{Portfolio}(t) - \max_{s \leq t} \text{Portfolio}(s)}{\max_{s \leq t} \text{Portfolio}(s)}
$$

---

## 📈 Activos Incluidos

| Ticker | Nombre | Asset Class | Retorno Esperado |
|--------|--------|-------------|------------------|
| **SPY** | S&P 500 ETF | Equity - Large Cap | 10% |
| **QQQ** | Nasdaq-100 ETF | Equity - Tech | 12% |
| **IWM** | Russell 2000 ETF | Equity - Small Cap | 8% |
| **EFA** | EAFE International | Equity - International | 7% |
| **EEM** | Emerging Markets | Equity - EM | 6% |
| **AGG** | Bond Aggregate | Fixed Income | 3% |
| **GLD** | Gold ETF | Commodities | 5% |
| **VNQ** | Real Estate ETF | Real Estate | 8% |
| **TLT** | Long-term Bonds | Fixed Income | 2% |
| **USO** | Oil ETF | Commodities | 0% |

---

## 🎓 Conceptos Clave

### Sharpe Ratio
Mide el exceso de retorno por unidad de riesgo:
- **> 1.0**: Excelente ratio riesgo-retorno
- **0.5 - 1.0**: Bueno
- **< 0.5**: Subóptimo

### Diversificación
El portfolio óptimo incluye solo 3 activos (SPY, QQQ, GLD) porque:
- Alta correlación entre activos similares
- SPY y QQQ ofrecen mejores retornos ajustados por riesgo
- GLD proporciona diversificación (baja correlación con equities)

### Frontera Eficiente
Portfolios en la frontera:
- **Izquierda**: Menor riesgo (más bonos)
- **Derecha**: Mayor retorno (más equities)
- **Punto óptimo**: Máximo Sharpe Ratio

---

## 📊 Comparación de Estrategias

| Estrategia | Retorno Anual | Volatilidad | Sharpe | Max DD |
|------------|---------------|-------------|--------|--------|
| **Portfolio Óptimo** | 20.38% | 10.79% | 1.61 | -10.26% |
| Equal Weight (10 activos) | 12.50% | 14.20% | 0.67 | -18.45% |
| 60/40 (Equity/Bonds) | 8.20% | 9.50% | 0.55 | -12.30% |
| 100% SPY | 10.50% | 18.00% | 0.42 | -23.50% |

**Conclusión**: El portfolio optimizado ofrece:
- **+63%** de retorno vs Equal Weight
- **-24%** menos volatilidad
- **+140%** mayor Sharpe Ratio

---

## 🔧 Tecnologías Utilizadas

| Categoría | Tecnologías |
|-----------|-------------|
| **Optimización** | scipy.optimize, numpy |
| **Análisis** | pandas, numpy |
| **Visualización** | Matplotlib, Seaborn, Plotly |
| **Dashboard** | Streamlit |
| **Data** | pandas, numpy |

---

## 🎯 Casos de Uso

### Asset Management
- ✅ Construcción de portfolios institucionales
- ✅ Rebalanceo automático
- ✅ Risk management

### Robo-Advisors
- ✅ Asignación automática de activos
- ✅ Perfiles de riesgo personalizados
- ✅ Optimización continua

### Wealth Management
- ✅ Portfolios para clientes de alto patrimonio
- ✅ Tax-loss harvesting
- ✅ Custom constraints

### Fondos de Inversión
- ✅ Diseño de fondos multi-asset
- ✅ Benchmark tracking
- ✅ Factor investing

---

## 📚 Extensiones Futuras

- [ ] **Black-Litterman Model** para incorporar views del mercado
- [ ] **Risk Parity** (equiponderación de riesgo)
- [ ] **Factor Models** (Fama-French 3/5 factores)
- [ ] **Rebalanceo automático** con triggers
- [ ] **Tax-aware optimization**
- [ ] **ESG constraints** (inversión sostenible)
- [ ] **Monte Carlo simulation** para proyecciones
- [ ] **Multi-period optimization**

---

## 📊 Resultados Visuales

### Frontera Eficiente
![Frontera Eficiente](../data/portfolio_analysis.png)

El gráfico muestra:
- **Estrella roja**: Portfolio de máximo Sharpe Ratio
- **Cuadrado azul**: Portfolio de mínima varianza
- **Puntos**: 5,000 portfolios simulados (color = Sharpe Ratio)

---

## 🎓 Referencias

### Papers Fundamentales
1. Markowitz, H. (1952). "Portfolio Selection"
2. Sharpe, W. (1964). "Capital Asset Pricing Model"
3. Black, F., & Litterman, R. (1991). "Global Portfolio Optimization"

### Libros Recomendados
- "Modern Portfolio Theory and Investment Analysis" - Elton et al.
- "Quantitative Portfolio Management" - Chincarini & Kim
- "Expected Returns" - Antti Ilmanen
