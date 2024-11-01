import numpy as np
from utils import MMQ, linear_system

#########################################################################
#____________________ IMPLEMENTACAO MACHINE LEARNING ____________________
#########################################################################

def train(ndig_treino: int, p: int) -> np.ndarray:
	"""
  Faz o treinamento para os dígitos de 0 a 9, armazenando os pesos 
	  em uma matriz tridimensional.
  
  Args:
    ndig_treino (int): Número de treinos para cada dígito, 
		  representando a quantidade de amostras para treinamento.
    p (int): Dimensão da matriz Wd, determinando o número 
		  de colunas na matriz de pesos resultante.
  
  Returns:
    np.ndarray: Uma matriz tridimensional W com dimensão [10, 784, p] 
      contendo os pesos do treinamento para cada dígito.
      Cada fatia W[d] contém a matriz de pesos para o dígito d.
  """

	W = []
	for digit in range(10):
		# nome do arquivo correspondente ao dígito
		filename = f"dados_mnist/train_dig{digit}.txt" 
		f = open(filename)
		fl = f.readlines()
		n = len(fl)
		A = []
		for i in range(n):
			fl[i] = fl[i].split(" ")
			for j in range(ndig_treino):
				fl[i][j] = int(fl[i][j])/255
			# A matriz A representa uma imagem e armazena apenas as colunas necessárias, com entradas int
			A.append(fl[i][0:ndig_treino]) 
		A = np.array(A)
		# W contém os treinamentos dos dígitos na forma fatorada (10 x 784 x p)
		W.append(MMQ(A, p))	
	W = np.array(W)
	return(W) 

#########################################################################	
def recognize(W: np.ndarray, ntest: int = 10000) -> None: 
	"""
  Realiza o reconhecimento de dígitos usando "test_images.txt".
  
  Para cada coluna da matriz de teste A, a função resolve o sistema 
  Wd * H = (coluna de A) para cada dígito (0 a 9), comparando o erro 
  entre as soluções para identificar o dígito correspondente ao menor erro.
  
  Args:
    W (np.ndarray): Matriz tridimensional de pesos com dimensão [10, 784, p], 
		  contendo as matrizes de fatoração Wd para cada dígito.
    ntest (int, opcional): Número de testes a serem realizados, com 
		  o padrão definido como 10000.
  
  Returns:
    None: A função imprime os resultados dos testes e não retorna valores.
  
  Observação:
    A função lê os dados de "test_images.txt". Para cada coluna de `A`, calcula 
		a solução que melhor aproxima cada dígito.
  """
	
	A = []
	f = open("dados_mnist/test_images.txt")
	fl = f.readlines()
	n = len(fl)
	for i in range(n):
			fl[i] = fl[i].split(" ")
			for j in range(ntest):
				fl[i][j] = int(fl[i][j])
			A.append(fl[i][0:ntest])
	# A contém o arquivo test_images.txt (784 x 10000)
	A = np.array(A)
	provavel = [0]*ntest     # vetor que armazena o dígito provável
	erro = [np.inf]*ntest    # vetor que armazena o erro relativo ao dígito
	for d in range(10):	
		Wd = np.copy(W[d])
		Ac = np.copy(A)
		H = linear_system(Wd, Ac)
		WdH = np.matmul(W[d],H)
		for j in range(ntest):
			soma = np.sum(np.subtract(A[0:n,j],WdH[0:n,j])**2)
			ej = np.sqrt(soma)
			if ej < erro[j]:
				erro[j] = ej
				provavel[j] = d 
	B=[]                 # B é um vetor com os verdadeiros dígitos
	d = [0]*10           # contagem dos dígitos de test_index
	acertos = [0]*10     # contagem dos acertos de cada dígito
	f = open("dados_mnist/test_index.txt")
	fl = f.readlines()
	for i in range(ntest):
		fl[i] = fl[i].split("\n")
		fl[i][0] = int(fl[i][0])
		B.append(fl[i][0])
		d[fl[i][0]] += 1
	for i in range(ntest):
		if B[i] == provavel[i]:
			acertos[B[i]] += 1
	# p_total = porcentagem total de acertos
	p_total = np.sum(acertos)/100 
	print("a) Porcentual total de acertos: " + str(p_total) + "%")	
	print("b) Porcentual de acertos de cada dígito: \n") 
	for i in range(10):
		# p_dig = porcentagem de acertos de cada dígito
		p_dig = np.round(100*acertos[i]/d[i],2) 
		print("Dígito " + str(i) + ":")
		print("Classificações corretas: " + str(acertos[i]))
		print("Percentual de acerto: " + str(p_dig) + "%")	
		print()

#########################################################################
def machine_learning(ndig_treino: int, p: int, n_test: int = 10000) -> None:
	"""
  Executa o treinamento e análise para reconhecimento de dígitos.
  
  A função chama o processo de treinamento para criar as matrizes de fatoração 
	e realiza a análise, testando o modelo treinado em um conjunto de dados de teste.
  
  Args:
    ndig_treino (int): Número de treinos para cada dígito, definindo o 
		  tamanho do conjunto de treino.
    p (int): Dimensão da matriz de fatoração utilizada no treinamento.
    n_test (int, opcional): Número de testes a serem realizados na fase 
		  de análise, com o valor padrão de 10000.
  
  Returns:
    None: A função executa o treinamento e a análise e imprime os resultados, sem retornar valores.
  """
	
	W = train(ndig_treino, p)
	recognize(W, n_test)
