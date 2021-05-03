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
from sklearn.model_selection import train_test_split, cross_val_score, cross_val_predict


#################################################################################
# Decisão se o código vai rodar como predição ou validação
#################################################################################
MODE_VALIDATION = False
# MODE_VALIDATION = True

#################################################################################
# Leitura dos arquivos de input
#################################################################################
def get_data(path):
    return pd.read_csv(path)

#################################################################################
# Preprocessamento
#################################################################################

#-------------------------------------------------------------------------------
# Elimincação das colunas não utilizadas
#-------------------------------------------------------------------------------
def filter_best_params(data, is_train):
    selected_params = [
        'produto_solicitado',
        'sexo',
        'idade',
        'estado_civil',
        'qtde_dependentes',
        'nacionalidade',
        'estado_onde_nasceu',
        'estado_onde_reside',
        'tipo_residencia',
        'meses_na_residencia',
        'renda_mensal_regular',
        'renda_extra',
        'qtde_contas_bancarias',
        'valor_patrimonio_pessoal',
        'possui_carro',
        'vinculo_formal_com_empresa',
        'possui_telefone_trabalho',
        'profissao',
        'ocupacao',
        'profissao_companheiro',
        'grau_instrucao_companheiro',
        'local_onde_reside',
        'local_onde_trabalha']

    output = ['inadimplente']

    if (is_train):
        selected_params.extend(output)
        return data[selected_params]
    
    return data[selected_params]

#-------------------------------------------------------------------------------
# Organização dos dados
#-------------------------------------------------------------------------------

# FORMATAÇÃO DOS DADOS
# --------------------------------------
def pretrain_format_data(data):
    data["sexo"].replace({" ": "N"}, inplace=True)
    data["estado_onde_nasceu"].replace({" ": "XX"}, inplace=True)
    data["tipo_residencia"].replace({np.nan: 6}, inplace=True)
    data["meses_na_residencia"].replace({np.nan: 0}, inplace=True)
    data["profissao"].replace({np.nan: 19}, inplace=True)
    data["ocupacao"].replace({np.nan: 6}, inplace=True)
    data["profissao_companheiro"].replace({np.nan: 18}, inplace=True)
    data["grau_instrucao_companheiro"].replace({np.nan: 6}, inplace=True)
    return data

# ALTERAÇÃO DE DADOS
# --------------------------------------
def pretrain_change_data(data):
    # # Organização do CEP nas regiões macros (a partir dos primeiros digitos)
    # for index in range(0, 10):
    #     data.loc[data["local_onde_reside"] // 100 == index , "local_onde_reside"] = index
    #     data.loc[data["local_onde_trabalha"] // 100 == index , "local_onde_trabalha"] = index

    # # Unir a informação "estado_onde_nasceu" e "estado_onde_reside" em uma coluna binarizada: se reside no estado em que nasceu
    # data['mora_onde_nasceu'] = np.where(data['estado_onde_nasceu'] == data['estado_onde_reside'], 1, 0)
    data = data.drop(['estado_onde_nasceu','estado_onde_reside'], axis=1)

    return data

# BINARIZAÇÃO
# --------------------------------------
def pretrain_data_binarizer(data):
    binarizer = LabelBinarizer()
    for param in ['possui_carro','vinculo_formal_com_empresa', 'possui_telefone_trabalho']:
        data[param] = binarizer.fit_transform(data[param])
    return data

# ONE-HOT ENCODING
# --------------------------------------
def pretrain_data_one_hot_encoding(data):
    data = pd.get_dummies(data,columns=['produto_solicitado'])
    data = pd.get_dummies(data,columns=['sexo'])
    data = pd.get_dummies(data,columns=['estado_civil'])
    data = pd.get_dummies(data,columns=['nacionalidade'])
    data = pd.get_dummies(data,columns=['tipo_residencia'])
    data = pd.get_dummies(data,columns=['profissao'])
    data = pd.get_dummies(data,columns=['ocupacao'])
    data = pd.get_dummies(data,columns=['profissao_companheiro'])
    data = pd.get_dummies(data,columns=['grau_instrucao_companheiro'])
    data = pd.get_dummies(data,columns=['local_onde_reside'])
    data = pd.get_dummies(data,columns=['local_onde_trabalha'])
    return data



#################################################################################
# Preparação dos dados para o treinamento
#################################################################################

#-------------------------------------------------------------------------------
# Embaralhamento dos dados
#-------------------------------------------------------------------------------
def shuffle_data(data):
    return data.sample(frac=1,random_state=12345)

#-------------------------------------------------------------------------------
# Remoção de dados que existem no treino e não existem no teste
#-------------------------------------------------------------------------------
def drop_difference_param_train_test(my_data, other_data):
    params = my_data.columns.difference(other_data.columns)
    params = params.to_numpy().tolist()

    print("\n\n", params, "\n\n")

    if ('inadimplente' in params):
        params.remove('inadimplente')
    return my_data.drop(params, axis=1)

#-------------------------------------------------------------------------------
# Adição de dados a mais que existem no treino e não existem no teste, e vice versa
#-------------------------------------------------------------------------------
def add_difference_param_train_test(my_data, other_data):
    params = my_data.columns.difference(other_data.columns)
    params = params.to_numpy().tolist()

    if ('inadimplente' in params):
        params.remove('inadimplente')

    for param in params:
        other_data[param] = [0] * (len(other_data.index))

    return other_data

# ------------------------------------------------------------------------------
# Divisão entre inputs e outputs
# ------------------------------------------------------------------------------
def split_inputs_outputs(data, param):
    x = data.loc[:,data.columns!=param].values
    y = data.loc[:,data.columns==param].values
    return x, y.ravel()

# ------------------------------------------------------------------------------
# Ajustar a escala dos atributos nos conjuntos de treino e de teste
# ------------------------------------------------------------------------------
def adjust_scale(data):
    scale_adjust = MinMaxScaler()
    scale_adjust.fit(data)
    return scale_adjust.transform(data)



#################################################################################
# Processamento: Treinamento e Predição
#################################################################################

#-------------------------------------------------------------------------------
# Treinamento do classificador KNN com o conjunto de treino
#-------------------------------------------------------------------------------
def train(x_train, y_train, _n_neighbors, _p):
    model = KNeighborsClassifier(
        n_neighbors = _n_neighbors,
        weights     = 'uniform',
        p           = _p)
    return model.fit(x_train,y_train)

#-------------------------------------------------------------------------------
# Predição do resultado com o classificador treinado
#-------------------------------------------------------------------------------
def predict(model, data):
    return model.predict(data)


#-------------------------------------------------------------------------------
# Validação do sistema com os dados usados (fazendo uso do treinamento cruzado)
#-------------------------------------------------------------------------------
def validation(x, y):
    print('\n\n\n')
    print ( "\nVALIDAÇÃO DO MODELO")
    print ( "\n  K   ACERTO(%)")
    print ( " --   ------")
    for k in range(31,120,2):
        classificator = KNeighborsClassifier(
            n_neighbors = k,
            weights     = 'uniform',
            p           = 1)

        scores = cross_val_score(classificator, x, y, cv=10)
        
        print ('k = %2d' % k, 'Acurácia média = %6.1f' % (100*sum(scores)/8))


#################################################################################
# Pós processamento
#################################################################################
#-------------------------------------------------------------------------------
# Exportar os resultados preditos em um arquivo ".csv" no formato correto
#-------------------------------------------------------------------------------
def export_to_csv(y_predict, start_id_count):
    # Cria a coluna com os indices corretos de saida
    y_id = np.array(np.arange(start_id_count, start_id_count+len(y_predict), 1).tolist())

    # Converte os arrays em dataframe
    DF = pd.DataFrame(data={
        'id_solicitante': y_id,
        'inadimplente': y_predict
        })
    
    # Salva o dataframe em um arquivo .csv sem a primeira coluna ser o indice padrão [0 a len(y_predict)]
    DF.to_csv("data.csv", index=False)



#################################################################################
# Funções gerais de lógica
#################################################################################
def preprocessing(data):
    data = pretrain_format_data(data)
    data = pretrain_change_data(data)
    data = pretrain_data_binarizer(data)
    data = pretrain_data_one_hot_encoding(data)
    data = shuffle_data(data)
    return data

def scoring(real, predict):
    score = sum(real==predict)/len(predict)
    print('\n\n\n')
    print("===> Acurácia do treino: %6.1f %%" % (100*score))

if __name__ == "__main__":
    # Le os dados dos arquivos e transforma em dataframes
    input_train_data = get_data('data\conjunto_de_treinamento.csv')
    input_test_data = get_data('data\conjunto_de_teste.csv')

    # Escolhe os mehlores parâmetros
    train_data = filter_best_params(input_train_data, True)
    test_data = filter_best_params(input_test_data, False)

    # Realiza toda a organização, formatação e configuração dos dados
    train_data = preprocessing(train_data)
    test_data = preprocessing(test_data)
    
    # Remove atributos que ou o treino nao tem, ou o teste
    # train_data = drop_difference_param_train_test(train_data, test_data)
    # test_data = drop_difference_param_train_test(test_data, train_data)
    test_data = add_difference_param_train_test(train_data, test_data)
    train_data = add_difference_param_train_test(test_data, train_data)

    # Split dos dados de input e outout do treinamento
    x_train, y_train = split_inputs_outputs(train_data, 'inadimplente')
    x_test = test_data

    # Normalizar dados
    x_train, x_test = adjust_scale(x_train), adjust_scale(x_test)

    if (MODE_VALIDATION):
        validation(x_train, y_train)
    else:
        # Treinamento
        k, p = 97, 1
        model_trained = train(x_train, y_train, k, p)

        # Predição
        y_predict_train = predict(model_trained, x_train)
        y_predict_test = predict(model_trained, x_test)

        # Indicação da acurácia do treino
        scoring(y_train, y_predict_train)

        # Exportação do resultado final da predição para a planilha de resposta
        export_to_csv(y_predict_test, 20001)