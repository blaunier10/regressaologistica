#!/usr/bin/env python
# coding: utf-8

# In[10]:


import streamlit as st
import pandas as pd
from sklearn.linear_model import LogisticRegression
import seaborn as sns 

# Carregar o modelo treinado e os dados
base = pd.read_excel('basedadosreg.xlsx')

caracteristica = base.iloc[:, 1:4].values
previsor = base.iloc[:, 4:5].values

# Treinar o modelo
funcao = LogisticRegression()
funcao.fit(caracteristica, previsor)

st.title('Previsão de Compra')

# Solicitar os dados do usuário
salario = st.number_input('Salário')
tipo_renda = st.selectbox('Tipo de Renda', ['Assalariado', 'Autônomo', 'Empresário'])
possui_imovel = st.selectbox('Possui Imóvel?', ['Sim', 'Não'])

# Mapear as opções para valores numéricos
tipo_renda_map = {'Assalariado': 1, 'Autônomo': 2, 'Empresário': 3}
possui_imovel_map = {'Sim': 1, 'Não': 2}

# Preparar os dados para fazer a previsão
tipo_renda_num = tipo_renda_map[tipo_renda]
possui_imovel_num = possui_imovel_map[possui_imovel]
parametro = [[salario, tipo_renda_num, possui_imovel_num]]

# Fazer a previsão
fazendo_previsao = funcao.predict(parametro)
probabilidade = funcao.predict_proba(parametro)

if fazendo_previsao == 0:
    st.write('Não vai comprar')
    st.write('Probabilidade:', probabilidade)
else:
    st.write('Vai comprar')
    st.write('Probabilidade:', probabilidade)
st.write("""
  Legenda das Probabilidades:
- Probabilidade de não comprar: Quanto mais próximo de 0, menor a chance de compra.
- Probabilidade de comprar: Quanto mais próximo de 1, maior a chance de compra.
""")

