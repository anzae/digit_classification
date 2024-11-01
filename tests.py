import numpy as np
from utils import Rot_givens, cos_sin, linear_system, MMQ

#########################################################################
#_____________________________ TESTES ___________________________________
#########################################################################

def teste_rot_givens() -> None:
	
# Teste da Rot-Givens do enunciado
# No teste, aplicamos a rotação em W e ele deve retornar a matriz rotacionada
# W = [[2,1, 1,-1, 1],       W_rot = [[2,1,        1,       -1,         1],
#      [0,3, 0, 1, 2],                [0,3,        0,        1,         2],
#      [0,0, 2, 2,-1],                [0,0,5/sqrt(5),3/sqrt(5),-4/sqrt(5)],
#      [0,0,-1, 1, 2],                [0,0,        0,4/sqrt(5), 3/sqrt(5)], 
#      [0,0, 0, 3, 1]]                [0,0,        0,         3,        1]]
	
	W = np.array([[2,1,1,-1,1],[0,3,0,1,2],[0,0,2,2,-1],[0,0,-1,1,2],[0,0,0,3,1]], dtype='f')
	c = cos_sin(W[2][2],W[3][2])
	Rot_givens(W,5,2,3,c[0],c[1])
	print("teste inicial: Rot-Givens")
	print("W = ")
	print(W)
	print()
	return()

#########################################################################
	
def teste_a() -> None:
	n=m=64
	W=np.zeros([n,m])
	b=np.zeros([n,1])
	for i in range(n):
		W[i][i]=2
		b[i][0]=1
		for j in range(n):
			if abs(i-j) == 1:
				W[i][j] = 1
	X = linear_system(W,b)
	print("teste a) ")
	print("X = ")
	print(X)
	print()
	return()

#########################################################################

def teste_b() -> None:
	n=20
	m=17
	W = np.zeros([n,m])
	b = np.zeros([n,1])
	for i in range(n):
		b[i][0] = i+1
		for j in range(m):
			if abs(i-j) <= 4:
				W[i][j] = 1/(i+j+1)
	# o enunciado pede 1/(i+j-1), mas como python os índices começam no 0
	# logo teremos 1/((i+1)+(j+1)-1) = 1/(i+j+1)
	X = linear_system(W,b)
	print("teste b) ")
	print("X = ")	
	print(X)
	print()
	return()

#########################################################################	

def teste_c() -> None:
	n=p=64
	m = 3
	W = np.zeros([n,p])
	A = np.zeros([n,m])
	for i in range(n):
		W[i][i] = 2
		A[i][0] = 1
		A[i][1] = i+1
		A[i][2] = 2*i+1
		for j in range(p):
			if abs(i-j) == 1:
				W[i][j] = 1		
	H = linear_system(W,A)
	print("teste c) ")
	print("X = ")
	print(H)
	print()
	return()

#########################################################################	

def teste_d() -> None:
	n=20
	p=17
	m=3
	W = np.zeros([n,p])
	A = np.zeros([n,m])
	for i in range(n):
		A[i][0] = 1
		A[i][1] = i+1
		A[i][2] = 2*i+1
		for j in range(p):
			if abs(i-j) <= 4:
				W[i][j] = 1/(i+j+1)
	X = linear_system(W,A)
	print("teste d) ")
	print("X = ")
	print(X)
	print()
	return()

#########################################################################

def tarefa_2() -> None:
	A = np.array([[3/10, 3/5, 0], [1/2, 0, 1], [4/10, 4/5, 0]], dtype = 'f')
	a = MMQ(A,2)
	print("Tarefa 2: fatorar a matriz A ")
	print("A = ")       # Inicialmente, a função MMQ estava definida para 
	print(A)            # retornar W, H e E(erro) em um vetor. Isso foi 
	print()             # alterado para a tarefa principal. A função MMQ  
	print("Matriz W: ") # retorna apenas a matriz W.
	print(a)            
	print()
	return()