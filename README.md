# Previsão do Preço do Diesel: Análise de Séries Temporais

![Python](https://img.shields.io/badge/Python-3.13+-3776AB?logo=python&logoColor=white)
![Plotly](https://img.shields.io/badge/Plotly-6.6.0-3F4F75?logo=plotly&logoColor=white)
![statsmodels](https://img.shields.io/badge/statsmodels-0.14+-4051B5)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.8+-F7931E?logo=scikitlearn&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-2.10.0-EE4C2C?logo=pytorch&logoColor=white)
![Optuna](https://img.shields.io/badge/Optuna-4.8.0-6DB3E8)

---

Análise e previsão do preço médio de revenda do diesel no Brasil (2013–hoje) a partir dos dados semanais da ANP, comparando três abordagens de modelagem: ARIMA, SVR e LSTM com otimização de hiperparâmetros via Optuna.

## Fonte de dados

**ANP — Agência Nacional do Petróleo, Gás Natural e Biocombustíveis**
Série histórica semanal do levantamento de preços de combustíveis:

```
https://www.gov.br/anp/pt-br/assuntos/precos-e-defesa-da-concorrencia/precos/precos-revenda-e-de-distribuicao-combustiveis/shlp/semanal/semanal-brasil-desde-2013.xlsx
```

- Frequência: semanal (segunda-feira)
- Cobertura: 459 municípios (26 capitais + 433 municípios do interior)
- Variável alvo: `PREÇO MÉDIO REVENDA` do `OLEO DIESEL` (R$/L)
- Pré-processamento: `asfreq("7D")` + interpolação linear para gaps

## Funções utilitárias

| Função | Descrição |
|--------|-----------|
| `calc_best_lags` | Identifica lags significativos via intervalo de confiança da ACF |
| `is_stationary` | Teste ADF com nível de significância configurável |
| `find_differencing_order` | Determina `d` aplicando ADF iterativamente até estacionariedade |
| `forecast` | Previsão online passo-a-passo (walk-forward) com ARIMA |
| `significant_lags_acf` | Retorna lista de lags fora do intervalo de confiança da ACF |
| `window_data` | Cria janelas deslizantes para modelos supervisionados (SVR/LSTM) |

## Pipeline

### 1. Pré-processamento

- Divisão cronológica 50/25/25: `train` / `val` / `test`
- Escalonamento `MinMaxScaler` fitado apenas no treino (sem data leakage)

### 2. Decomposição

Análise manual da série em três componentes, com duas abordagens para a tendência:

| Método | Componente |
|--------|------------|
| Média Móvel (52 semanas) | Tendência |
| Suavização Exponencial (`α=0.2`) | Tendência |
| Padrão sazonal médio por posição | Sazonalidade |
| Resíduo = Série − Tendência − Sazonalidade | Resíduo |

Decomposição automática também disponível via `seasonal_decompose` (statsmodels, `period=52`).

### 3. Modelagem

#### ARIMA

Seleção de parâmetros via ACF/PACF e teste ADF:

```python
p = 1   # PACF — ordem autorregressiva
d = 1   # diferenciação para estacionariedade
q = 1   # ACF — ordem da média móvel
```

Previsão com **walk-forward**: o modelo é reajustado a cada passo incorporando o valor real observado, evitando acúmulo de erro de longo prazo.

#### SVR

- Janela deslizante: `window_size = 45` semanas
- Busca em grade com `TimeSeriesSplit(n_splits=5)`:

```python
param_grid = {
    "C":       [0.1, 1, 10, 100],
    "epsilon": [0.01, 0.1, 0.5],
    "kernel":  ["rbf", "linear"],
}
```

#### LSTM

Arquitetura: `LSTM → Activation → Linear`

Otimização de hiperparâmetros com **Optuna** (TPE Sampler, `n_trials=15`):

| Hiperparâmetro | Espaço de busca |
|----------------|-----------------|
| `hidden_size` | {64, 96} |
| `num_layers` | 1 – 3 |
| `dropout` | 0.0 – 0.2 |
| `epochs` | {100, 200} |
| `activation` | relu / tanh |
| `optimizer` | adam / sgd |
| `lr` | 1e-4 – 1e-2 (log) |

Janelas avaliadas: `[45, 26, 14, 13, 1]` semanas — permite comparar o impacto da memória histórica na qualidade da previsão.

Suporte a **AMP** (Automatic Mixed Precision) quando CUDA disponível.


### Checkpoints históricos
 
Eventos relevantes sobrepostos automaticamente em todos os gráficos de série temporal, ancorados no valor da curva (não fixos no topo):
 
| # | Data | Tipo | Evento |
|---|------|------|--------|
| *1 | Out/2016 | 🇧🇷 Política | Petrobras adota PPI — preços seguem Brent + câmbio |
| *2 | 27/05–26/06/2018 | 🇧🇷 Nacional | Greve dos caminhoneiros — vendas de diesel caem ~30% |
| *3 | Mar–Abr/2020 | 🌍 Externo | COVID-19 + guerra de preços Rússia-Arábia Saudita — Brent cai >65%; WTI atinge −US$37 |
| *4 | 24/02/2022 | 🌍 Externo | Invasão russa da Ucrânia — Brent dispara para US$133 (+30% em 2 semanas) |
| *5 | Out/2022 | 🌍 Externo | Corte OPEP+ de ~2 mb/dia — reversão de queda nos preços |
| *6 | 16/05/2023 | 🇧🇷 Política | Petrobras abandona PPI (gov. Lula) — queda imediata de ~R$0,44/L |
| *7 | Ago/2023 | 🇧🇷 Política | Petrobras reajusta diesel +25,8% e gasolina +16,2% |
| *7 | Mar/2026 | 🇧🇷 Guerra | Ataque dos EUA & Israel mata lider do Irã |

 
## Dependências
 
```
uv sync
```
 
Ou manualmente:
 
```
pip install pandas numpy plotly statsmodels scikit-learn torch optuna kaleido
```
 
## Referências
 
1. [ANP — Série Histórica de Preços de Combustíveis](https://www.gov.br/anp/pt-br/assuntos/precos-e-defesa-da-concorrencia/precos/precos-revenda-e-de-distribuicao-combustiveis/serie-historica-do-levantamento-de-precos)
2. Hyndman, R.J. & Athanasopoulos, G. — *Forecasting: Principles and Practice* (3ª ed.)
3. [Optuna: A hyperparameter optimization framework](https://optuna.org)
 