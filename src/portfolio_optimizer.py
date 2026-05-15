import pandas as pd
import numpy as np
from scipy.optimize import minimize
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import joblib

np.random.seed(42)

class PortfolioOptimizer:
    def __init__(self, tickers, start_date='2020-01-01', end_date='2024-12-31'):
        self.tickers = tickers
        self.start_date = start_date
        self.end_date = end_date
        self.prices = None
        self.returns = None
        self.mean_returns = None
        self.cov_matrix = None
        self.optimal_weights = None

    def generate_synthetic_data(self):
        """Generar datos sintéticos de precios de activos"""
        print("Generando datos historicos sinteticos...")

        # Parámetros realistas de mercado
        market_params = {
            'SPY': {'mu': 0.10, 'sigma': 0.18},   # S&P 500 ETF
            'QQQ': {'mu': 0.12, 'sigma': 0.22},   # Nasdaq ETF
            'IWM': {'mu': 0.08, 'sigma': 0.24},   # Russell 2000
            'EFA': {'mu': 0.07, 'sigma': 0.20},   # EAFE International
            'EEM': {'mu': 0.06, 'sigma': 0.28},   # Emerging Markets
            'AGG': {'mu': 0.03, 'sigma': 0.06},   # Bond Aggregate
            'GLD': {'mu': 0.05, 'sigma': 0.16},   # Gold
            'VNQ': {'mu': 0.08, 'sigma': 0.22},   # Real Estate
            'TLT': {'mu': 0.02, 'sigma': 0.12},   # Long-term Bonds
            'USO': {'mu': 0.00, 'sigma': 0.35}    # Oil
        }

        # Generar fechas de trading (excluir fines de semana)
        dates = pd.date_range(start=self.start_date, end=self.end_date, freq='B')
        n_days = len(dates)

        # Inicializar precios
        prices_dict = {}

        for ticker in self.tickers:
            if ticker in market_params:
                params = market_params[ticker]
            else:
                # Parámetros por defecto
                params = {'mu': 0.08, 'sigma': 0.20}

            # Retornos diarios (ajustados)
            daily_mu = params['mu'] / 252
            daily_sigma = params['sigma'] / np.sqrt(252)

            # Generar retornos con GBM (Geometric Brownian Motion)
            returns = np.random.normal(daily_mu, daily_sigma, n_days)

            # Precio inicial
            price_initial = 100

            # Calcular precios
            prices = [price_initial]
            for ret in returns:
                prices.append(prices[-1] * (1 + ret))

            prices_dict[ticker] = prices[1:]  # Excluir precio inicial

        # Crear DataFrame
        self.prices = pd.DataFrame(prices_dict, index=dates)
        self.prices.index.name = 'Date'

        print(f"Datos generados: {len(self.prices)} dias, {len(self.tickers)} activos")
        return self

    def calculate_returns(self):
        """Calcular retornos"""
        print("\nCalculando retornos...")
        self.returns = self.prices.pct_change().dropna()
        self.mean_returns = self.returns.mean()
        self.cov_matrix = self.returns.cov()

        print(f"Retorno promedio anual:")
        for ticker in self.tickers:
            annual_ret = self.mean_returns[ticker] * 252 * 100
            print(f"  {ticker}: {annual_ret:.2f}%")

        return self

    def portfolio_stats(self, weights):
        """Calcular estadísticas del portfolio"""
        # Retorno anualizado
        portfolio_return = np.sum(self.mean_returns * weights) * 252

        # Volatilidad anualizada
        portfolio_std = np.sqrt(np.dot(weights.T, np.dot(self.cov_matrix * 252, weights)))

        return portfolio_return, portfolio_std

    def negative_sharpe_ratio(self, weights, risk_free_rate=0.03):
        """Sharpe Ratio negativo (para minimización)"""
        p_return, p_std = self.portfolio_stats(weights)
        sharpe = (p_return - risk_free_rate) / p_std
        return -sharpe

    def optimize_sharpe(self, risk_free_rate=0.03):
        """Optimizar para máximo Sharpe Ratio"""
        print("\n" + "="*60)
        print("OPTIMIZACION: MAXIMO SHARPE RATIO")
        print("="*60)

        n_assets = len(self.tickers)

        # Constraints: suma de pesos = 1
        constraints = {'type': 'eq', 'fun': lambda x: np.sum(x) - 1}

        # Bounds: 0 <= peso <= 1 (sin short selling)
        bounds = tuple((0, 1) for _ in range(n_assets))

        # Pesos iniciales (equal weighted)
        init_weights = np.array([1/n_assets] * n_assets)

        # Optimizar
        result = minimize(
            self.negative_sharpe_ratio,
            init_weights,
            args=(risk_free_rate,),
            method='SLSQP',
            bounds=bounds,
            constraints=constraints
        )

        optimal_weights = result.x
        p_return, p_std = self.portfolio_stats(optimal_weights)
        sharpe = (p_return - risk_free_rate) / p_std

        print(f"\nPesos optimos:")
        for ticker, weight in zip(self.tickers, optimal_weights):
            if weight > 0.01:  # Solo mostrar si >1%
                print(f"  {ticker}: {weight*100:.2f}%")

        print(f"\nRetorno esperado: {p_return*100:.2f}%")
        print(f"Volatilidad (riesgo): {p_std*100:.2f}%")
        print(f"Sharpe Ratio: {sharpe:.4f}")

        self.optimal_weights_sharpe = optimal_weights
        return self

    def optimize_min_variance(self):
        """Optimizar para mínima varianza"""
        print("\n" + "="*60)
        print("OPTIMIZACION: MINIMA VARIANZA")
        print("="*60)

        n_assets = len(self.tickers)

        def portfolio_variance(weights):
            return np.dot(weights.T, np.dot(self.cov_matrix * 252, weights))

        constraints = {'type': 'eq', 'fun': lambda x: np.sum(x) - 1}
        bounds = tuple((0, 1) for _ in range(n_assets))
        init_weights = np.array([1/n_assets] * n_assets)

        result = minimize(
            portfolio_variance,
            init_weights,
            method='SLSQP',
            bounds=bounds,
            constraints=constraints
        )

        optimal_weights = result.x
        p_return, p_std = self.portfolio_stats(optimal_weights)

        print(f"\nPesos optimos (minima varianza):")
        for ticker, weight in zip(self.tickers, optimal_weights):
            if weight > 0.01:
                print(f"  {ticker}: {weight*100:.2f}%")

        print(f"\nRetorno esperado: {p_return*100:.2f}%")
        print(f"Volatilidad (riesgo): {p_std*100:.2f}%")

        self.optimal_weights_minvar = optimal_weights
        return self

    def efficient_frontier(self, num_portfolios=5000):
        """Calcular frontera eficiente"""
        print("\nCalculando frontera eficiente...")

        n_assets = len(self.tickers)
        results = np.zeros((3, num_portfolios))

        for i in range(num_portfolios):
            # Pesos aleatorios
            weights = np.random.random(n_assets)
            weights /= np.sum(weights)

            # Estadísticas
            p_return, p_std = self.portfolio_stats(weights)
            sharpe = (p_return - 0.03) / p_std

            results[0,i] = p_return
            results[1,i] = p_std
            results[2,i] = sharpe

        self.efficient_frontier_results = results
        print(f"Frontera eficiente calculada: {num_portfolios} portfolios")
        return self

    def calculate_risk_metrics(self, weights, confidence_level=0.95):
        """Calcular métricas de riesgo: VaR y CVaR"""
        # Simular retornos del portfolio
        portfolio_returns = (self.returns @ weights)

        # VaR (Value at Risk)
        var = np.percentile(portfolio_returns, (1 - confidence_level) * 100)

        # CVaR (Conditional VaR / Expected Shortfall)
        cvar = portfolio_returns[portfolio_returns <= var].mean()

        # Máximo Drawdown
        cumulative_returns = (1 + portfolio_returns).cumprod()
        running_max = cumulative_returns.expanding().max()
        drawdown = (cumulative_returns - running_max) / running_max
        max_drawdown = drawdown.min()

        return {
            'VaR_95': var * 100,
            'CVaR_95': cvar * 100,
            'Max_Drawdown': max_drawdown * 100
        }

    def backtest(self, weights, initial_capital=100000):
        """Backtesting del portfolio"""
        print("\n" + "="*60)
        print("BACKTESTING")
        print("="*60)

        # Retornos del portfolio
        portfolio_returns = (self.returns @ weights)

        # Valor del portfolio en el tiempo
        portfolio_value = initial_capital * (1 + portfolio_returns).cumprod()

        # Métricas
        total_return = (portfolio_value.iloc[-1] / initial_capital - 1) * 100
        annual_return = ((portfolio_value.iloc[-1] / initial_capital) ** (252 / len(portfolio_returns)) - 1) * 100
        annual_vol = portfolio_returns.std() * np.sqrt(252) * 100

        # Métricas de riesgo
        risk_metrics = self.calculate_risk_metrics(weights)

        print(f"\nCapital inicial: ${initial_capital:,.2f}")
        print(f"Capital final: ${portfolio_value.iloc[-1]:,.2f}")
        print(f"Retorno total: {total_return:.2f}%")
        print(f"Retorno anualizado: {annual_return:.2f}%")
        print(f"Volatilidad anualizada: {annual_vol:.2f}%")
        print(f"Sharpe Ratio: {(annual_return - 3) / annual_vol:.4f}")

        print(f"\nMetricas de Riesgo:")
        print(f"  VaR (95%): {risk_metrics['VaR_95']:.2f}%")
        print(f"  CVaR (95%): {risk_metrics['CVaR_95']:.2f}%")
        print(f"  Max Drawdown: {risk_metrics['Max_Drawdown']:.2f}%")

        self.backtest_results = {
            'portfolio_value': portfolio_value,
            'portfolio_returns': portfolio_returns,
            'total_return': total_return,
            'annual_return': annual_return,
            'annual_vol': annual_vol,
            'risk_metrics': risk_metrics
        }

        return self

    def plot_results(self):
        """Generar gráficos"""
        print("\nGenerando graficos...")

        fig = plt.figure(figsize=(16, 12))

        # 1. Frontera Eficiente
        ax1 = plt.subplot(2, 3, 1)
        scatter = ax1.scatter(
            self.efficient_frontier_results[1,:] * 100,
            self.efficient_frontier_results[0,:] * 100,
            c=self.efficient_frontier_results[2,:],
            cmap='viridis',
            s=10,
            alpha=0.5
        )
        plt.colorbar(scatter, label='Sharpe Ratio')

        # Portfolio óptimo (max Sharpe)
        p_return, p_std = self.portfolio_stats(self.optimal_weights_sharpe)
        ax1.scatter(p_std*100, p_return*100, color='red', s=200, marker='*',
                   edgecolors='black', linewidths=2, label='Max Sharpe', zorder=5)

        # Portfolio mínima varianza
        p_return_mv, p_std_mv = self.portfolio_stats(self.optimal_weights_minvar)
        ax1.scatter(p_std_mv*100, p_return_mv*100, color='blue', s=200, marker='s',
                   edgecolors='black', linewidths=2, label='Min Variance', zorder=5)

        ax1.set_xlabel('Volatilidad (%)')
        ax1.set_ylabel('Retorno Esperado (%)')
        ax1.set_title('Frontera Eficiente')
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # 2. Asset Allocation (Max Sharpe)
        ax2 = plt.subplot(2, 3, 2)
        weights_to_plot = [(t, w) for t, w in zip(self.tickers, self.optimal_weights_sharpe) if w > 0.01]
        tickers_plot, weights_plot = zip(*weights_to_plot) if weights_to_plot else ([], [])

        ax2.pie(weights_plot, labels=tickers_plot, autopct='%1.1f%%', startangle=90)
        ax2.set_title('Asset Allocation (Max Sharpe)')

        # 3. Performance en el tiempo
        ax3 = plt.subplot(2, 3, 3)
        portfolio_value = self.backtest_results['portfolio_value']
        ax3.plot(portfolio_value.index, portfolio_value.values, linewidth=2)
        ax3.set_xlabel('Fecha')
        ax3.set_ylabel('Valor del Portfolio ($)')
        ax3.set_title('Performance del Portfolio')
        ax3.grid(True, alpha=0.3)
        ax3.ticklabel_format(style='plain', axis='y')

        # 4. Drawdown
        ax4 = plt.subplot(2, 3, 4)
        cumulative_returns = (1 + self.backtest_results['portfolio_returns']).cumprod()
        running_max = cumulative_returns.expanding().max()
        drawdown = (cumulative_returns - running_max) / running_max * 100

        ax4.fill_between(drawdown.index, drawdown.values, 0, color='red', alpha=0.3)
        ax4.plot(drawdown.index, drawdown.values, color='red', linewidth=1)
        ax4.set_xlabel('Fecha')
        ax4.set_ylabel('Drawdown (%)')
        ax4.set_title('Portfolio Drawdown')
        ax4.grid(True, alpha=0.3)

        # 5. Distribución de retornos
        ax5 = plt.subplot(2, 3, 5)
        portfolio_returns_pct = self.backtest_results['portfolio_returns'] * 100
        ax5.hist(portfolio_returns_pct, bins=50, edgecolor='black', alpha=0.7)
        ax5.axvline(portfolio_returns_pct.mean(), color='red', linestyle='--',
                   linewidth=2, label=f'Media: {portfolio_returns_pct.mean():.2f}%')
        ax5.axvline(self.backtest_results['risk_metrics']['VaR_95'], color='orange',
                   linestyle='--', linewidth=2, label=f"VaR 95%: {self.backtest_results['risk_metrics']['VaR_95']:.2f}%")
        ax5.set_xlabel('Retorno Diario (%)')
        ax5.set_ylabel('Frecuencia')
        ax5.set_title('Distribucion de Retornos')
        ax5.legend()
        ax5.grid(True, alpha=0.3)

        # 6. Correlación de activos
        ax6 = plt.subplot(2, 3, 6)
        corr_matrix = self.returns.corr()
        sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap='coolwarm',
                   center=0, square=True, ax=ax6, cbar_kws={'shrink': 0.8})
        ax6.set_title('Matriz de Correlacion')

        plt.tight_layout()
        plt.savefig('../data/portfolio_analysis.png', dpi=300, bbox_inches='tight')
        print("Graficos guardados en: data/portfolio_analysis.png")

        return self

    def save_results(self):
        """Guardar resultados"""
        print("\nGuardando resultados...")

        # Guardar precios y retornos
        self.prices.to_csv('../data/historical_prices.csv')
        self.returns.to_csv('../data/returns.csv')

        # Guardar pesos óptimos
        results = {
            'optimal_weights_sharpe': dict(zip(self.tickers, self.optimal_weights_sharpe)),
            'optimal_weights_minvar': dict(zip(self.tickers, self.optimal_weights_minvar)),
            'backtest_metrics': {
                'total_return': self.backtest_results['total_return'],
                'annual_return': self.backtest_results['annual_return'],
                'annual_vol': self.backtest_results['annual_vol'],
                'sharpe_ratio': (self.backtest_results['annual_return'] - 3) / self.backtest_results['annual_vol'],
                **self.backtest_results['risk_metrics']
            }
        }

        pd.DataFrame([results]).to_json('../data/optimization_results.json', orient='records')

        print("Resultados guardados")
        return self


def main():
    # Activos a optimizar
    tickers = ['SPY', 'QQQ', 'IWM', 'EFA', 'EEM', 'AGG', 'GLD', 'VNQ', 'TLT', 'USO']

    print("="*60)
    print("OPTIMIZACION DE PORTFOLIO - TEORIA MODERNA (MARKOWITZ)")
    print("="*60)

    optimizer = PortfolioOptimizer(tickers)

    optimizer.generate_synthetic_data() \
             .calculate_returns() \
             .optimize_sharpe() \
             .optimize_min_variance() \
             .efficient_frontier() \
             .backtest(optimizer.optimal_weights_sharpe) \
             .plot_results() \
             .save_results()

    print("\n" + "="*60)
    print("OPTIMIZACION COMPLETADA")
    print("="*60)


if __name__ == "__main__":
    main()
