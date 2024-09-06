import numpy as np
import yfinance as yf
from keras.models import load_model
from sklearn.preprocessing import MinMaxScaler

# Função para carregar dados do Yahoo Finance
def yahoo_to_dataset(ticker, period='5y'):
    # Baixando dados de histórico do Yahoo Finance
    data = yf.download(ticker, period=period)
    
    # Normalizando os dados de OHLCV (Open, High, Low, Close, Volume)
    ohlcv_histories = data[['Open', 'High', 'Low', 'Close', 'Volume']].values
    scaler = MinMaxScaler()
    ohlcv_histories = scaler.fit_transform(ohlcv_histories)
    
    # Calculando indicadores técnicos (neste exemplo, vamos usar a média móvel simples)
    data['SMA'] = data['Close'].rolling(window=15).mean()
    data['SMA'] = scaler.fit_transform(data[['SMA']].fillna(0))
    technical_indicators = data[['SMA']].values
    
    # Pegando o próximo valor de abertura como a variável de saída (preço de abertura do próximo dia)
    next_day_open_values = data['Open'].shift(-1).fillna(method='ffill').values.reshape(-1, 1)
    unscaled_y = data['Open'].shift(-1).fillna(method='ffill').values.reshape(-1, 1)
    
    # Normalizando o preço de abertura para ser usado na previsão
    y_normaliser = MinMaxScaler()
    next_day_open_values = y_normaliser.fit_transform(next_day_open_values)
    
    return ohlcv_histories, technical_indicators, next_day_open_values, unscaled_y, y_normaliser

# Carregando o modelo treinado
model = load_model('technical_model.h5')

# Carregando dados da ação usando a função yahoo_to_dataset
ticker = 'B3SA3.SA'
ohlcv_histories, technical_indicators, next_day_open_values, unscaled_y, y_normaliser = yahoo_to_dataset(ticker)

# Definindo a proporção de treino/teste
test_split = 0.9
n = int(ohlcv_histories.shape[0] * test_split)

# Dividindo os dados em treino e teste
ohlcv_train = ohlcv_histories[:n]
tech_ind_train = technical_indicators[:n]
y_train = next_day_open_values[:n]

ohlcv_test = ohlcv_histories[n:]
tech_ind_test = technical_indicators[n:]
y_test = next_day_open_values[n:]

unscaled_y_test = unscaled_y[n:]

# Fazendo as previsões com o modelo treinado
y_test_predicted = model.predict([ohlcv_test, tech_ind_test])

# Revertendo a normalização para obter os valores em escala real
y_test_predicted = y_normaliser.inverse_transform(y_test_predicted)

# Inicializando listas para armazenar as operações de compra e venda
buys = []
sells = []
thresh = 0.1  # Definindo um limite para quando comprar ou vender

# Definindo o intervalo de análise
start = 0
end = -1

# Iterando sobre o conjunto de teste para determinar operações de compra e venda
x = -1
for ohlcv, ind in zip(ohlcv_test[start: end], tech_ind_test[start: end]):
    normalised_price_today = ohlcv[-1][0]  # Preço normalizado do dia atual
    normalised_price_today = np.array([[normalised_price_today]])
    
    # Revertendo a normalização para o preço real do dia
    price_today = y_normaliser.inverse_transform(normalised_price_today)
    
    # Prevendo o preço de abertura do próximo dia
    predicted_price_tomorrow = np.squeeze(y_normaliser.inverse_transform(model.predict([[ohlcv], [ind]])))
    
    # Calculando a diferença entre o preço previsto e o preço atual
    delta = predicted_price_tomorrow - price_today
    
    # Adicionando operação de compra ou venda com base no delta
    if delta > thresh:
        buys.append((x, price_today[0][0]))
    elif delta < -thresh:
        sells.append((x, price_today[0][0]))
    
    x += 1

# Exibindo a quantidade de operações de compra e venda
print(f"buys: {len(buys)}")
print(f"sells: {len(sells)}")

# Função para calcular os ganhos com base nas operações de compra e venda
def compute_earnings(buys_, sells_):
    purchase_amt = 10  # Definindo o valor de compra
    stock = 0  # Quantidade de ações
    balance = 0  # Saldo em dinheiro
    while len(buys_) > 0 and len(sells_) > 0:
        if buys_[0][0] < sells_[0][0]:  # Verificando se há uma compra antes da venda
            balance -= purchase_amt  # Subtraindo o valor de compra do saldo
            stock += purchase_amt / buys_[0][1]  # Calculando a quantidade de ações compradas
            buys_.pop(0)
        else:
            balance += stock * sells_[0][1]  # Calculando o valor de venda e atualizando o saldo
            stock = 0  # Zerando a quantidade de ações após a venda
            sells_.pop(0)
    print(f"earnings: ${balance}")  # Exibindo o saldo final

# Calculando os ganhos com as operações de compra e venda realizadas
compute_earnings([b for b in buys], [s for s in sells])

# Plotando os gráficos de preços reais e previstos
import matplotlib.pyplot as plt

plt.gcf().set_size_inches(22, 15, forward=True)

# Plotando o preço real e o preço previsto
real = plt.plot(unscaled_y_test[start:end], label='real')
pred = plt.plot(y_test_predicted[start:end], label='predicted')

# Adicionando os pontos de compra (verde) e venda (vermelho) no gráfico
if len(buys) > 0:
    plt.scatter(list(list(zip(*buys))[0]), list(list(zip(*buys))[1]), c='#00ff00', s=50)
if len(sells) > 0:
    plt.scatter(list(list(zip(*sells))[0]), list(list(zip(*sells))[1]), c='#ff0000', s=50)

plt.legend(['Real', 'Predicted', 'Buy', 'Sell'])

# Exibindo o gráfico final
plt.show()
