# Previsão de Preços de Ações Usando Redes Neurais LSTM

Este projeto utiliza uma rede neural LSTM (Long Short-Term Memory) para prever o preço de abertura do próximo dia de uma ação com base em dados históricos de mercado, incluindo indicadores técnicos. A previsão é feita utilizando dados obtidos diretamente da API do Yahoo Finance e o modelo treinado previamente.

### Colabore Conosco!

Curte nossos conteúdos? Então inscreva-se no nosso canal do YouTube e acompanhe de perto todas as novidades, dicas e atualizações que preparamos com muito carinho para você! Basta clicar aqui: [Tudo Mais Constante](https://www.youtube.com/channel/UCVOAEiukuYC2rnNO9ZCNHqQ?sub_confirmation=1).

E não para por aí! No Instagram, estamos sempre postando conteúdos exclusivos e interagindo com nossa comunidade. Siga-nos agora mesmo em [@tmconstante](https://instagram.com/tmconstante) e fique por dentro de tudo!

## Objetivo

O objetivo deste projeto é realizar previsões de preços de ações para identificar potenciais pontos de compra e venda com base nas previsões da rede neural. Este projeto simula operações de compra e venda para calcular o saldo final com base nas previsões feitas pelo modelo.

## Funcionalidades

- Download de dados de ações diretamente do Yahoo Finance via a biblioteca `yfinance`.
- Processamento de dados de mercado e cálculo de indicadores técnicos.
- Previsão de preços usando um modelo LSTM previamente treinado (armazenado em `technical_model.h5`).
- Identificação de pontos de compra e venda com base nas previsões de preço do próximo dia.
- Simulação de operações de compra e venda para calcular o saldo final.
- Visualização dos preços reais e previstos com indicação dos pontos de compra/venda.

## Estrutura do Código

1. **Download dos Dados**: Utilizamos a API do Yahoo Finance para baixar dados históricos da ação, como preço de abertura, fechamento, volume, etc.
2. **Preprocessamento**: Os dados são normalizados e processados para serem usados como entrada no modelo LSTM.
3. **Modelo LSTM**: O modelo LSTM previamente treinado é carregado e usado para fazer previsões dos preços de abertura do próximo dia.
4. **Identificação de Oportunidades de Compra/Venda**: O script avalia a diferença entre o preço previsto e o preço real para decidir se deve "comprar" ou "vender".
5. **Cálculo de Ganhos**: Uma simulação de compra e venda é feita para calcular o saldo final.
6. **Visualização**: Um gráfico é gerado para visualizar o desempenho das previsões em relação aos preços reais, além de marcar as compras e vendas no gráfico.

## Como Usar

### Pré-requisitos

- Python 3.x
- Bibliotecas Python necessárias:
  - `tensorflow` (para carregar e executar o modelo LSTM)
  - `numpy`
  - `yfinance` (para baixar os dados do Yahoo Finance)
  - `matplotlib` (para plotar os gráficos)
  - `scikit-learn` (para normalização dos dados)

### Instalação

1. Clone o repositório:

   ```bash
   git clone https://github.com/seu-usuario/nome-do-repositorio.git


### Contribuição

Se você deseja contribuir para este projeto, sinta-se à vontade para fazer um fork e abrir uma pull request com suas alterações. 
Sugestões de melhorias também são bem-vindas!


