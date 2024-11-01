# Classificador de Dígitos com MNIST

## Descrição
Este projeto implementa um classificador de dígitos utilizando dados extraídos da base de dados MNIST. O objetivo é treinar classificadores para reconhecer dígitos de 0 a 9 através de técnicas de decomposição de matrizes. O projeto inclui uma interface de usuário implementada com PySimpleGUI para facilitar a execução dos testes.

## Dados Utilizados
Os dados foram extraídos da base de dados MNIST. Já foram pré-processados e estão disponíveis nos seguintes arquivos, dentro da pasta `dados_mnist`:
- `train_dig0.txt` até `train_dig9.txt`: Dados de treinamento contendo imagens de dígitos.
- `test_images.txt`: Contém 10.000 imagens de dígitos para testes.
- `test_index.txt`: Contém os índices verdadeiros dos dígitos presentes nas imagens de teste.

## Dependências
Este projeto foi desenvolvido em Python 3 e requer as seguintes bibliotecas:
- NumPy
- PySimpleGUI

Para instalar as dependências, utilize o seguinte comando:
```pip3 install numpy pysimplegui```

## Estrutura de Arquivos
- `README.md`: Este arquivo contendo informações sobre o projeto.
- `main.py`: Arquivo principal executável.
- `learning.py`: Arquivo que contém a implementação do classificador.
- `utils.py`: Arquivo contendo funções auxiliares para manipulação de matrizes.
- `tests.py`: Arquivo contendo os testes inicial, 1-a, 1-b, 1-c, 1-d e 2, solicitados no enunciado.
- `dados_mnist`: Pasta contendo os seguintes arquivos: 
  - `train_dig{n}.txt`: Arquivos de treinamento para dígitos de 0 a 9.
  - `test_images.txt`: Arquivo contendo as imagens de teste.
  - `test_index.txt`: Arquivo contendo os índices verdadeiros das imagens de teste.
