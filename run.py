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
import pandas   as pd
import numpy    as np
import seaborn  as sns; sns.set()
from sklearn.neighbors          import KNeighborsClassifier
from sklearn.preprocessing      import LabelBinarizer, MinMaxScaler, PolynomialFeatures
from matplotlib                 import pyplot as plt
from sklearn.model_selection    import train_test_split, cross_val_score, GridSearchCV
from sklearn.ensemble           import RandomForestClassifier, RandomForestRegressor, GradientBoostingRegressor, AdaBoostRegressor
from sklearn.metrics            import confusion_matrix, accuracy_score, precision_score, r2_score
from sklearn.metrics            import recall_score, f1_score, roc_auc_score, mean_squared_error
from sklearn.metrics            import plot_precision_recall_curve, plot_roc_curve
from sklearn.linear_model       import LinearRegression, Ridge, Lasso, SGDRegressor
from sklearn.svm                import SVR
from sklearn.feature_selection  import mutual_info_regression

#################################################################################
# Decisão se o código vai rodar como predição ou validação
#################################################################################
# MODE_VALIDATION = True
# MODE_CROSS_VALIDATION = False

# MODE_VALIDATION = False
# MODE_CROSS_VALIDATION = True

MODE_VALIDATION = False
MODE_CROSS_VALIDATION = False

#################################################################################
# Leitura dos arquivos de input
#################################################################################
def get_data(path):
    return pd.read_csv(path)

#################################################################################
# Preprocessamento
#################################################################################

#-------------------------------------------------------------------------------
# Visualização da correlação dos parâmetros usados
#-------------------------------------------------------------------------------
def show_correlation_matrix(data):
    print("\n\n")
    print("MATRIZ DE CORRELAÇÃO:")
    corrMatrix = data.corr()
    sns.heatmap(corrMatrix, xticklabels=1, yticklabels=1, vmin=0, vmax=1)
    plt.show()
    print("\n\n")

#-------------------------------------------------------------------------------
# Elimincação das colunas não utilizadas
#-------------------------------------------------------------------------------
def filter_best_params(data, is_train):
    selected_params = [
        'produto_solicitado',
        'dia_vencimento',
        'forma_envio_solicitacao',
        'tipo_endereco',
        'sexo',
        'idade',
        'estado_civil',
        'qtde_dependentes',
        'nacionalidade',
        'estado_onde_nasceu',
        'estado_onde_reside',
        'possui_telefone_residencial',
        # 'codigo_area_telefone_residencial',
        'tipo_residencia',
        'meses_na_residencia',
        'possui_email',
        'renda_mensal_regular',
        'renda_extra',
        'possui_cartao_visa',
        'possui_cartao_mastercard',
        'possui_cartao_diners',
        'possui_cartao_amex',
        'possui_outros_cartoes',
        'qtde_contas_bancarias',
        'valor_patrimonio_pessoal',
        'possui_carro',
        'vinculo_formal_com_empresa',
        # 'estado_onde_trabalha',
        'possui_telefone_trabalho',
        # 'codigo_area_telefone_trabalho',
        'meses_no_trabalho',
        'profissao',
        'ocupacao',
        'profissao_companheiro',
        'grau_instrucao_companheiro',
        'local_onde_reside',
        'local_onde_trabalha'
        ]

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
    data["estado_onde_nasceu"].replace({" ": data["estado_onde_nasceu"].mode()}, inplace=True)
    data["tipo_residencia"].replace({np.nan: int(data["tipo_residencia"].mode())}, inplace=True)
    data["meses_na_residencia"].replace({np.nan: int(data["meses_na_residencia"].mode())}, inplace=True)
    data["profissao"].replace({np.nan: int(data["profissao"].mode())}, inplace=True)
    data["ocupacao"].replace({np.nan: int(data["ocupacao"].mode())}, inplace=True)
    data["profissao_companheiro"].replace({np.nan: int(data["profissao_companheiro"].mode())}, inplace=True)
    data["grau_instrucao_companheiro"].replace({np.nan: int(data["profissao_companheiro"].mode())}, inplace=True)
    return data

# ALTERAÇÃO DE DADOS
# --------------------------------------
def pretrain_change_data(data):
    # Organização do CEP nas regiões macros (a partir dos primeiros digitos)
    # for index in range(0, 10):
    #     data.loc[data["local_onde_reside"] // 100 == index , "local_onde_reside"] = index
    #     data.loc[data["local_onde_trabalha"] // 100 == index , "local_onde_trabalha"] = index

    # # Unir a informação "estado_onde_nasceu" e "estado_onde_reside" em uma coluna binarizada: se reside no estado em que nasceu
    data['mora_onde_nasceu'] = np.where(data['estado_onde_nasceu'] == data['estado_onde_reside'], 1, 0)
    data = data.drop(['estado_onde_nasceu','estado_onde_reside'], axis=1)

    return data

# BINARIZAÇÃO
# --------------------------------------
def pretrain_data_binarizer(data):
    binarizer = LabelBinarizer()
    for param in ['possui_telefone_residencial','vinculo_formal_com_empresa', 'possui_carro', 'possui_telefone_trabalho']:
        data[param] = binarizer.fit_transform(data[param])
    return data

# ONE-HOT ENCODING
# --------------------------------------
def pretrain_data_one_hot_encoding(data):
    one_hot_encoding_params = [
        'produto_solicitado',
        'sexo',
        'estado_civil',
        'nacionalidade',
        'tipo_residencia',
        'profissao',
        'ocupacao',
        'profissao_companheiro',
        'grau_instrucao_companheiro',
        'local_onde_reside',
        'local_onde_trabalha',
        'forma_envio_solicitacao',
        'tipo_endereco',
        'dia_vencimento'
        ]
    data = pd.get_dummies(data,columns=one_hot_encoding_params)
    return data

def pretrain_categorical_data_formater(data):
    one_hot_encoding_params = [
        'produto_solicitado',
        'sexo',
        'estado_civil',
        'nacionalidade',
        'tipo_residencia',
        'profissao',
        'ocupacao',
        'profissao_companheiro',
        'grau_instrucao_companheiro',
        'local_onde_reside',
        'local_onde_trabalha',
        'forma_envio_solicitacao',
        'tipo_endereco',
        'dia_vencimento'
        ]
    for label, content in data.items():
        if (label in one_hot_encoding_params):
            data[label] = pd.Categorical(content).codes+1
    return data

# Move a coluna target para a ultima coluna
# --------------------------------------
def move_overdue_to_end(data, target_col):
    overdue_col = data.pop(target_col)
    data.insert(len(data.T), target_col, overdue_col)
    return data


#################################################################################
# Preparação dos dados para o treinamento
#################################################################################

#-------------------------------------------------------------------------------
# Embaralhamento dos dados
#-------------------------------------------------------------------------------
def shuffle_data(data):
    return data.sample(frac=1,random_state=0)

#-------------------------------------------------------------------------------
# Remoção de dados que existem no treino e não existem no teste
#-------------------------------------------------------------------------------
def drop_difference_param_train_test(my_data, other_data):
    params = my_data.columns.difference(other_data.columns)
    params = params.to_numpy().tolist()
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
    x = data.loc[:,data.columns!=param]
    y = data.loc[:,data.columns==param]
    return x, y

def concat_train_test(train, test):
    return pd.concat([train, test], ignore_index=True)

def split_train_test(data, row):
    train = data.loc[:row-1,:]
    test = data.loc[row:,:]
    return train, test

# ------------------------------------------------------------------------------
# Ajustar a escala dos atributos nos conjuntos de treino e de teste
# ------------------------------------------------------------------------------
def adjust_scale(data):
    scale_adjust = MinMaxScaler()
    scale_adjust.fit(data)
    data[data.columns] = scale_adjust.transform(data[data.columns])
    return data



#################################################################################
# Processamento: Treinamento e Predição
#################################################################################

#-------------------------------------------------------------------------------
# Treinamento do classificador com o conjunto de treino
#-------------------------------------------------------------------------------
def train_KNN(x_train, y_train, _n_neighbors, _p):
    model = KNeighborsClassifier(
        n_neighbors = _n_neighbors,
        weights     = 'uniform',
        p           = _p)
    return model.fit(x_train,y_train)

def train_Random_Forest(x_train, y_train, depth):
    model = RandomForestClassifier(
            max_depth=depth, 
            random_state=0)
    return model.fit(x_train,y_train)

def adjust_params_Polynomial_Regression(x, deg):
    pf = PolynomialFeatures(degree=deg)
    return pf.fit_transform(x)

def train_Linear_Regression(x_train, y_train):
    model = LinearRegression()
    return model.fit(x_train, y_train)

def train_Lasso(x_train, y_train, alp):
    model = Lasso(
            alpha=alp, 
            random_state=0)
    return model.fit(x_train,y_train)

def train_Ridge(x_train, y_train, alp):
    model = Ridge(
            alpha=alp, 
            random_state=0)
    return model.fit(x_train,y_train)

def train_SGD(x_train, y_train, alp, tolerance):
    model = SGDRegressor(
            alpha=alp, 
            loss='squared_loss', # squared_loss, huber, epsilon_insensitive, squared_epsilon_insensitive
            penalty='elasticnet', # l2, l1, elasticnet
            tol=tolerance,
            max_iter=100000,
            random_state=0)
    return model.fit(x_train,y_train)

def train_Random_Forest_R(x_train, y_train, depth):
    model = RandomForestRegressor(
            max_depth=depth,
            # max_features='log2',
            # min_samples_split=5,
            # min_samples_leaf=2,
            # min_weight_fraction_leaf=0.2, 
            random_state=0)
    return model.fit(x_train,y_train)

def train_GridSearchCV(x_train, y_train):
    parameters = {'kernel':('linear', 'rbf'), 'C':[1, 10]}
    svr = SVR()
    model = GridSearchCV(svr, parameters)
    return model.fit(x_train,y_train)

def train_GradientBoostingRegressor(x_train, y_train):
    model = GradientBoostingRegressor(random_state=0)
    return model.fit(x_train,y_train)

def train_AdaBoostRegressor(x_train, y_train):
    model = AdaBoostRegressor(
        random_state=0,
        n_estimators=50,
        learning_rate=1.0,
        loss='exponential' # linear, square, exponential
        )
    return model.fit(x_train,y_train)

#-------------------------------------------------------------------------------
# Predição do resultado com o classificador treinado
#-------------------------------------------------------------------------------
def predict(model, data):
    y_predict = model.predict(data)
    size = len(y_predict)
    y = [0]*size

    if (y_predict.dtype == np.dtype('float64')):
        threshold = 0.5
        for i in range(size):
            y[i] = 1 if (y_predict[i] >= threshold) else 0
        y = np.array(y)
    else:
        y = y_predict

    return y

#-------------------------------------------------------------------------------
# Validação do sistema com os dados usados (fazendo uso do treinamento cruzado)
#-------------------------------------------------------------------------------
def validation(x, y):
    print('\n\n\n')
    print ( "\nVALIDAÇÃO DO MODELO")

    x_train, x_test, y_train, y_test = train_test_split(
        x,
        y,
        test_size = 0.1,
        random_state = 0   
        )

    # --------
    # Treinamento com KNN
    # k, p = 97, 1
    # model_trained = train_KNN(x_train, y_train, k, p)

    # Treinamento com Random Forest
    depth = 25
    model_trained = train_Random_Forest(x_train, y_train, depth)

    # Treinamento com Regressão com regularização Lasso
    # degree = 2
    # x_train = adjust_params_Polynomial_Regression(x_train, degree)
    # x_test = adjust_params_Polynomial_Regression(x_test, degree)
    # Regressão linear
    # model_trained = train_Linear_Regression(x_train, y_train)
    # Lasso
    # alpha = 0.001
    # model_trained = train_Lasso(x_train, y_train, alpha)
    # Ridge
    # alpha = 0.001
    # model_trained = train_Ridge(x_train, y_train, alpha)
    # SGD
    # alpha, tolerance = 0.001, 1e-6
    # model_trained = train_SGD(x_train, y_train, alpha, tolerance)
    
    # --------

    # Predição
    y_predict_train = predict(model_trained, x_train)
    y_predict_test = predict(model_trained, x_test)

    # Indicação da acurácia do treino
    # scoring(y_train, y_predict_train)
    # scoring(y_test, y_predict_test)

    show_metrics(y_test, y_predict_test)
    rmse, r2 = get_error_metrics (y_test, y_predict_test)
    print('\n Depth =%2d  RMSE = %2.4f  R2 = %2.4f' % (depth, rmse, r2))
        
def cross_validation_KNN(x, y):
    print('\n\n\n')
    print ( "\nVALIDAÇÃO CRUZADA DO MODELO")
    print ( "\n  K   ACERTO(%)")
    print ( " --   ------")
    cross_val = 5
    for k in range(11,20,2):
        classificator = KNeighborsClassifier(
            n_neighbors = k,
            weights     = 'uniform',
            p           = 1)

        scores = cross_val_score(classificator, x, y, cv=cross_val)
        
        print ('k = %2d' % k, 'Acurácia média = %6.1f' % (100*sum(scores)/cross_val))
        
def cross_validation_Random_Forest(x, y):
    print('\n\n\n')
    print ( "\nVALIDAÇÃO CRUZADA DO MODELO")
    print ( "\n  D   ACERTO(%)")
    print ( " --   ------")

    cross_val = 5

    for depth in range(1,16):
        classificator = train_Random_Forest_R(x, y, depth)

        scores = cross_val_score(classificator, x, y, cv=cross_val, scoring='neg_root_mean_squared_error')
        
        print (' %2d' % depth, 'RMSE = %6.1f' % (100*sum(scores)/cross_val))
        
def cross_validation_Polynomial_Regression(x, y):
    print('\n\n\n')
    print ( "\nVALIDAÇÃO CRUZADA DO MODELO")
    print ( "\n  G   ACERTO(%)")
    print ( " --   ------")

    for degree in range(1,3):
        x = adjust_params_Polynomial_Regression(x, degree)
        # Regressão linear
        # model = train_Linear_Regression(x, y)
        # Lasso
        # alpha = 0.001
        # model = train_Lasso(x, y, alpha)
        # Ridge
        # alpha = 0.001
        # model = train_Ridge(x, y, alpha)
        # SGD
        alpha, tolerance = 0.001, 1e-6
        model = train_SGD(x, y, alpha, tolerance)

        y_pred = predict(model, x)
        rmse, r2 = get_error_metrics (y, y_pred)
        print('\n Degree =%2d  RMSE = %2.4f  R2 = %2.4f' % (degree, rmse, r2))


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

def show_metrics(y_true, y_pred):
    tp, fn, fp, tn = confusion_matrix(y_pred,y_true,labels=[True, False]).reshape(4,1)
    print (" Matriz de Confusão : TP=%4d | FN=%4d | FP=%4d | TN=%4d" % (tp, fn, fp, tn))
    print ("  " )
    print (" Acurácia           : %6.1f %%" % (100*accuracy_score(y_pred,y_true)) )
    print ("  " )   
    print (" Precisão           : %6.1f %%" % (100*precision_score(y_pred,y_true)) )
    print ("  " )   
    print (" Sensibilidade      : %6.1f %%" % (100*recall_score(y_pred,y_true)) )
    print ("  " )   
    print (" Score F1           : %6.1f %%" % (100*f1_score(y_pred,y_true)) )
    print ("  " )   
    print (" Área sob ROC       : %6.1f %%" % (100*roc_auc_score(y_pred,y_true)) )


def get_error_metrics (y_true, y_pred):
    mse  = mean_squared_error(y_pred, y_true)
    rmse = np.sqrt(mse)
    r2   = r2_score(y_pred, y_true)
    return rmse, r2

def plot_curves(model, x_test, y_pred):
    plot_roc_curve(model, x_test, y_pred)
    plot_precision_recall_curve(model, x_test, y_pred)

#################################################################################
# Funções gerais de lógica
#################################################################################
def preprocessing(data):
    data = pretrain_format_data(data)
    data = pretrain_change_data(data)
    data = pretrain_data_binarizer(data)
    data = pretrain_data_one_hot_encoding(data)
    # data = shuffle_data(data)
    return data

def scoring(real, predict):
    score = sum(real==predict)/len(predict)
    ## print('\n\n\n')
    print("===> Acurácia: %6.1f %%" % (100*score))

if __name__ == "__main__":
    pd.set_option("mode.chained_assignment", None)

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

    # Alinha todos os parametros em ordem alfabetica
    train_data = train_data.reindex(sorted(train_data.columns), axis=1)
    test_data = test_data.reindex(sorted(test_data.columns), axis=1)

    # Move a coluna de inadimplente para a ultima coluna
    train_data = move_overdue_to_end(train_data, 'inadimplente')

    # Mostra a relação entre os parâmetros
    # show_correlation_matrix(train_data)

    # Split dos dados de input e outout do treinamento
    x_train, y_train = split_inputs_outputs(train_data, 'inadimplente')
    x_test = test_data
    y_train = y_train.values.ravel()

    # ---
    x_train_rows = len(x_train)
    x_train_0, x_test_0 = x_train, x_test
    data = concat_train_test(x_train, x_test)
    # data = pretrain_categorical_data_formater(data)
    data_0 = data.copy()
    data = adjust_scale(data)
    x_train, x_test = split_train_test(data, x_train_rows)

    # Obtenção dos parametros que influenciam no resultado final
    mutual_info = mutual_info_regression(x_train, y_train, random_state=42, n_neighbors=10)
    mutual_info = pd.Series(mutual_info)
    mutual_info.index = x_train.columns
    mutual_info = mutual_info.sort_values(ascending=False)
    # mutual_info.plot.bar(figsize=(15,5))
    # Seleção dos parametros que influenciam no resultado final
    selected_top_columns = []
    for x in range(len(mutual_info)):
        if (mutual_info[x] > 0.0):
            selected_top_columns.append(mutual_info.index[x])
    x_train = x_train[selected_top_columns]
    x_test = x_test[selected_top_columns]

    if (MODE_VALIDATION):
        validation(x_train, y_train)
    elif (MODE_CROSS_VALIDATION):
        # cross_validation_KNN(x_train, y_train)
        cross_validation_Random_Forest(x_train, y_train)
        # cross_validation_Polynomial_Regression(x_train, y_train)
    else:
        # Treinamento com KNN
        # k, p = 100, 1
        # model_trained = train_KNN(x_train, y_train, k, p)
        # Treinamento com Random Forest
        # depth = 10
        # model_trained = train_Random_Forest(x_train, y_train, depth)
        # Treinamento com Regressão Polinomial
        # degree = 2
        # x_train = adjust_params_Polynomial_Regression(x_train, degree)
        # x_test = adjust_params_Polynomial_Regression(x_test, degree)
        # Regressão linear
        # model_trained = train_Linear_Regression(x_train, y_train)
        # Lasso
        # alpha = 0.001
        # model_trained = train_Lasso(x_train, y_train, alpha)
        # Ridge
        # alpha = 0.001
        # model_trained = train_Ridge(x_train, y_train, alpha)
        # SGD
        # alpha, tolerance = 0.01, 1e-8
        # model_trained = train_SGD(x_train, y_train, alpha, tolerance)

        depth = 7
        model_trained = train_Random_Forest_R(x_train, y_train, depth)
        # model_trained = train_GridSearchCV(x_train, y_train)
        # model_trained = train_GradientBoostingRegressor(x_train, y_train)
        # model_trained = train_AdaBoostRegressor(x_train, y_train)

        # Predição
        y_predict_train = predict(model_trained, x_train)
        y_predict_test = predict(model_trained, x_test)

        # Indicação da acurácia do treino
        #scoring(y_train, y_predict_train)
        
        # Indicação das métricas
        show_metrics(y_train, y_predict_train)
        rmse, r2 = get_error_metrics (y_train, y_predict_train)
        print('\n RMSE = %2.4f  R2 = %2.4f' % (rmse, r2))
        # plot_curves(model_trained, x_test, y_predict_test)

        # Exportação do resultado final da predição para a planilha de resposta
        export_to_csv(y_predict_test, 20001)