#################################################################################
# Universidade Federal do Rio de Janeiro
# Disciplina: Introdução ao Aprendizado de Máquina - EEL891
# Professor: Heraldo L. S. Almeida
# Desenvolvedor: Chritian Marques de Oliveira Silva
# DRE: 117.214.742
# Trabalho 1: Classificação - Sistema de apoio à decisão p/ aprovação de crédito
#################################################################################

#################################################################################
# Importação de bibliotecas
#################################################################################
import pandas as pd
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelBinarizer, MinMaxScaler
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split, cross_val_score


#################################################################################
# Leitura dos arquivos de input
#################################################################################
input_train_data = pd.read_csv('data\conjunto_de_treinamento.csv')
# input_test_data_x = pd.read_csv('data\conjunto_de_teste.csv')
# input_test_data_y = pd.read_csv('data\exemplo_arquivo_respostas.csv')

#################################################################################
# Preprocessamento
#################################################################################

# --------------------------------------
# Elimincação das colunas não utilizadas
# --------------------------------------
train_data = input_train_data.drop([
    'id_solicitante',
    'dia_vencimento',
    'forma_envio_solicitacao',
    'tipo_endereco',
    'grau_instrucao',
    'possui_telefone_residencial',
    'codigo_area_telefone_residencial',
    'possui_telefone_celular',
    'possui_email',
    'possui_cartao_visa',
    'possui_cartao_mastercard',
    'possui_cartao_diners',
    'possui_cartao_amex',
    'possui_outros_cartoes',
    'qtde_contas_bancarias_especiais',
    'estado_onde_trabalha',
    'codigo_area_telefone_trabalho',
    'meses_no_trabalho']
    , axis=1)

# Sobrou:
#       produto_solicitado
#       sexo
#       idade
#       estado_civil
#       qtde_dependentes
#       nacionalidade
#       estado_onde_nasceu
#       estado_onde_reside
#       tipo_residencia
#       meses_na_residencia
#       renda_mensal_regular
#       renda_extra
#       qtde_contas_bancarias
#       valor_patrimonio_pessoal
#       possui_carro
#       vinculo_formal_com_empresa
#       possui_telefone_trabalho
#       profissao
#       ocupacao
#       profissao_companheiro
#       grau_instrucao_companheiro
#       local_onde_reside
#       local_onde_trabalha
#       inadimplente


# --------------------------------------
# Organização dos dados
# --------------------------------------
# FORMATAÇÃO DOS DADOS
# --------------------------------------
train_data["sexo"].replace({" ": "N"}, inplace=True)
train_data["estado_onde_nasceu"].replace({" ": "XX"}, inplace=True)
train_data["tipo_residencia"].replace({np.nan: 6}, inplace=True)
train_data["meses_na_residencia"].replace({np.nan: 0}, inplace=True)
train_data["profissao"].replace({np.nan: 19}, inplace=True)
train_data["ocupacao"].replace({np.nan: 6}, inplace=True)
train_data["profissao_companheiro"].replace({np.nan: 18}, inplace=True)
train_data["grau_instrucao_companheiro"].replace({np.nan: 6}, inplace=True)



# ALTERAÇÃO DE DADOS
# --------------------------------------
# Organização do CEP nas regiões macros (a partir dos primeiros digitos)
for index in range(0, 10):
    train_data.loc[train_data["local_onde_reside"] // 100 == index , "local_onde_reside"] = index
    train_data.loc[train_data["local_onde_trabalha"] // 100 == index , "local_onde_trabalha"] = index
# Unir a informação "estado_onde_nasceu" e "estado_onde_reside" em uma coluna binarizada: se reside no estado em que nasceu
train_data['mora_onde_nasceu'] = np.where(train_data['estado_onde_nasceu'] == train_data['estado_onde_reside'], 1, 0)
train_data = train_data.drop(['estado_onde_nasceu','estado_onde_reside'], axis=1)

# BINARIZAÇÃO
# --------------------------------------
binarizador = LabelBinarizer()
for param in ['possui_carro','vinculo_formal_com_empresa', 'possui_telefone_trabalho']:
    train_data[param] = binarizador.fit_transform(train_data[param])

# Substitui em "sexo" N=0, F=1 e M=2 
train_data['sexo_'] = np.where(train_data['sexo'] == "N", 0, 1)
train_data.loc[train_data["sexo"] == "F" , "sexo_"] = 1
train_data.loc[train_data["sexo"] == "M" , "sexo_"] = 2
train_data = train_data.drop(['sexo'], axis=1)
train_data.rename(columns={'sexo_': 'sexo'}, inplace=True)

# ONE-HOT ENCODING
# --------------------------------------
# train_data = pd.get_dummies(train_data,columns=['produto_solicitado'])
# train_data = pd.get_dummies(train_data,columns=['sexo'])
# train_data = pd.get_dummies(train_data,columns=['estado_civil'])
# train_data = pd.get_dummies(train_data,columns=['nacionalidade'])
# train_data = pd.get_dummies(train_data,columns=['tipo_residencia'])
# train_data = pd.get_dummies(train_data,columns=['profissao'])
# train_data = pd.get_dummies(train_data,columns=['ocupacao'])
# train_data = pd.get_dummies(train_data,columns=['profissao_companheiro'])
# train_data = pd.get_dummies(train_data,columns=['grau_instrucao_companheiro'])

# train_data = pd.get_dummies(train_data,columns=['local_onde_reside'])
# train_data = pd.get_dummies(train_data,columns=['local_onde_trabalha'])

x = train_data.loc[:,train_data.columns!='inadimplente'].values
y = train_data.loc[:,train_data.columns=='inadimplente'].values