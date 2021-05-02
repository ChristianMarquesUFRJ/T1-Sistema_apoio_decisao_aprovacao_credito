# Sistema de apoio à aprovação de Crédito
Trabalho 1 de Introdução ao Aprendizado de Máquina. Tem por objetivo a criação de um classificador para apoio à decisão de aprovação de crédito.

## 1. Pré-processamento

Analisando os dados e o que cada item significa, foi compreendida melhor as colunas e o que cada uma impacta no resultado de inadimplência. Também notou-se quais colunas poderiam deixar de existir (por não impactar diretamente o resultado), juntar-s

## 2. Seleção de atributos


| **ID da coluna** | Título | Categoria |
| :--- | :---: | :---: | :---: |
| 01 | produto_solicitado| one-hot |
| 05 | sexo | one-hot |
| 06 | idade | - |
| 07 | estado_civil | one-hot |
| 08 | qtde_dependentes | - |
| 10 | nacionalidade | one-hot |
| 11 e 12 | estado_onde_nasceu e estado_onde_reside | binarização |
| 15 | tipo_residencia | one-hot |
| 16 | meses_na_residencia | - |
| 19 | renda_mensal_regular | - |
| 20 | renda_extra | - |
| 26 | qtde_contas_bancarias | - |
| 28 | valor_patrimonio_pessoal | - |
| 29 | possui_carro | binarização |
| 30 | vinculo_formal_com_empresa | binarização |
| 32 | possui_telefone_trabalho | binarização |
| 35 | profissao | one-hot |
| 36 | ocupacao | one-hot |
| 37 | profissao_companheiro | one-hot |
| 38 | grau_instrucao_companheiro | one-hot |
| 39 | local_onde_reside | one-hot# |
| 40 | local_onde_trabalha | one-hot# |
| 41 | inadimplente | - |

#: Agupar valores nas 5 regiões (S, SE, CO, N e NE).

## 3. Implementação
### 3.1. Escolha do modelo preditivo
### 3.2. Script
### 3.3. Ajustes e hiperparâmetros
## 4. Análise geral do desempenho
