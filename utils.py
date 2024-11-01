import numpy as np

#########################################################################
#____________________________ FUNÇÕES ___________________________________
#########################################################################

def cos_sin(wik: float, wjk: float) -> tuple[float, float]: 
	"""
  Calcula os valores cosseno e seno de acordo com os escalares W[i,k] e W[j,k].
  
  Args:
    wik (float): Um número real representando W[i,k].
    wjk (float): Um número real representando W[j,k].
  
  Returns:
    Tuple[float, float]: Uma tupla contendo (cos, sin).
  """
	
	if np.abs(wik)>np.abs(wjk):
		tau = -np.divide(wjk, wik)
		c = 1 / np.sqrt(tau**2 + 1)
		s = c * tau
	else:
		tau = -np.divide(wik, wjk)
		s = 1 / np.sqrt(tau**2 + 1)
		c = s * tau
	return(c, s)
	
#########################################################################		

def Rot_givens(W: np.ndarray, m: int, i: int, j: int, c: float, s: float) -> None: 
	"""
  Aplica uma rotação de Givens na matriz W, rotacionando 
  as linhas i e j com os coeficientes de rotação c e s.
  
  Args:
    W (np.ndarray): Uma matriz de dimensão [n, m] (número 
      de linhas e número de colunas).
    m (int): Número de colunas da matriz W.
    i (int): Índice da primeira linha que será rotacionada.
    j (int): Índice da segunda linha que será rotacionada.
    c (Union[float, np.float64]): Valor do cosseno para a rotação de Givens.
    s (Union[float, np.float64]): Valor do seno para a rotação de Givens.
  
  Returns:
    None: A função altera a matriz W diretamente e não possui retorno.
  
  Observação:
    Esta função utiliza a rotina de rotação de Givens nº 5 apresentada.
  """
	
	W[i, 0:m], W[j, 0:m] = np.subtract(np.multiply(c, W[i,0:m]), np.multiply(s, W[j,0:m])), np.subtract(np.multiply(s, W[i,0:m]), -np.multiply(c, W[j,0:m]))
	return()

#########################################################################

def linear_system(W: np.ndarray, b: np.ndarray) -> np.ndarray:
	"""
  Resolve um sistema linear WX = b utilizando rotações de Givens 
	para escalonar a matriz W.

  Args:
    W (np.ndarray): Matriz de entrada[n, m], onde n é o número 
		  de equações e m é o número de variáveis.
    b (np.ndarray): Matriz de entrada[n, p], onde n é o número 
		  de equações e p é o número de colunas do lado direito.

  Returns:
    np.ndarray: A matriz X[m, p], solução do sistema WX = b.
  
  Observação:
    A função aplica rotações de Givens para escalonar a matriz W.
  """

	n = len(W)
	m = len(W[0])
	p = len(b[0])
	X = np.zeros([m,p])
	for k in range(m):
		for j in range(n-1, k, -1):
			if W[j, k] != 0:
				cos, sin = cos_sin(W[j-1,k], W[j,k]) # aqui usamos i = j-1
				Rot_givens(W, m, j-1, j, cos, sin) 
				Rot_givens(b, p, j-1, j, cos, sin) 
	for e in range(p):
		X[m-1, e] = np.divide(b[m-1, e],W[m-1, m-1])
		for k in range(m-2, -1, -1):
			# soma += W[k,j]*X[j,e] , k<=j<m
			soma = np.sum(np.multiply(W[k, k:m],X[k:m, e]))
			# X[k,e] = (b[k,e]-soma)/W[k,k]
			X[k, e] = np.divide(np.subtract(b[k, e], soma), W[k, k])
	return(X)
	
#########################################################################

def error(A: np.ndarray, W: np.ndarray, H: np.ndarray) -> float:
	"""
  Calcula o erro quadrático entre a matriz `A` e o produto das matrizes `W` e `H`, 
  definido como E = ||A - WH||^2 .
  
  Args:
    A (np.ndarray): Matriz original de dimensão [n, m].
    W (np.ndarray): Matriz fatorada de dimensão [n, p].
    H (np.ndarray): Matriz fatorada de dimensão [p, m].
  
  Returns:
    float: O valor escalar do erro quadrático E.
  """
	
	n = len(A)
	m = len(A[0])
	Q = np.matmul(W, H) 
	soma = 0
	# soma += (A[i,j] - Q[i,j])^2, com 0<=i<n e 0<=j<m
	soma = np.sum([np.sum(np.power(np.subtract(A[0:n, 0:m],Q[0:n, 0:m]), 2)), soma])
	return(soma)

#########################################################################

def MMQ(A: np.ndarray, p: int) -> np.ndarray:
	"""
  Realiza a fatoração da matriz A usando o método dos mínimos quadrados, 
	decompondo-a em duas matrizes W e H, onde A = WH.
  Busca minimizar o erro quadrático quando a fatoração exata não é possível.
  
  Args:
    A (np.ndarray): Matriz de entrada com dimensão [n, m], onde n é 
		  o número de linhas e m é o número de colunas.
    p (int): Dimensão da matriz de fatoração, determinando a quantidade 
		  de colunas em W e a quantidade de linhas em H.
  
  Returns:
    np.ndarray: A matriz W com dimensão [n, p], que é uma das matrizes de fatoração de A.
  """

	n = len(A)
	m = len(A[0])
	eps = np.power(0.1, 5)
	itmax = 100
	W = np.random.random([n, p]) # inicializa W aleatoriamente
	E = 2
	E_ant = 1
	it = 0
	while abs(E-E_ant)/E_ant > eps and it < itmax: # erro relativo
		A_copy = np.copy(A)
		A_t = np.copy(np.transpose(A))
		it += 1 # nº de iterações
		for k in range(p):
			# normalização da matriz W por colunas
			W[0:n, k] = np.divide(W[0:n, k], np.sqrt(np.sum(np.power(W[0:n, k], 2))))
		H = linear_system(W, A_copy) 		
		H[0:p, 0:m] = np.maximum(0, H[0:p, 0:m])
		H_t = np.copy(np.transpose(H))
		W_t = linear_system(H_t, A_t)
		W = np.transpose(W_t)
		W[0:n,0:p] = np.maximum(0, W[0:n, 0:p])
		E_ant = E
		E = error(A, W, H)
	return(W)