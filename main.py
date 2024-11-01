import numpy as np
import PySimpleGUI as sg

from tests import teste_rot_givens, teste_a, teste_b, teste_c, teste_d, tarefa_2
from learning import machine_learning

#########################################################################
#____________________________ INTERFACE _________________________________
#########################################################################

# Para simplificar a análise dos testes, foi implementado um layout com
# a biblioteca PySimpleGUI, utilizando botões para cada um dos testes.

layout = [[sg.Text('Testes', auto_size_text=True)], 
          [sg.Text('Teste inicial')],         
          [sg.Button('Rot-Givens')], 
		  [sg.Text('Primeira tarefa - Testes')],
		  [sg.Button('A', size=(2,1)), sg.Button('B',size=(2,1)), sg.Button('C',size=(2,1)), sg.Button('D',size=(2,1))],
		  [sg.Text('Segunda tarefa - Fatoração')],
		  [sg.Button('A = WH')],
		  [sg.Text('Tarefa principal')],
		  [sg.Button('Machine Learning')],
		  [sg.Button('Sair')]]

window = sg.Window('Atividade Machine Learning', layout) 

def machine() -> None:
	"""
    Abre uma janela para o usuário selecionar os inputs `ndig_treino` e `p`,
    e executa o reconhecimento de dígitos com os parâmetros fornecidos.
    
    A função chama `machine_learning` para realizar o treinamento e a análise de reconhecimento.
    """

	p = 0
	ndig_treino = 0
	layout = [[sg.Text('Tarefa principal: Machine Learning')],      
            [sg.Text('ndig_treino: ')], 
			      [sg.Radio('100', 'treino', size=(10, 1)), sg.Radio('1000', 'treino', size=(10, 1)), sg.Radio('4000', 'treino', size=(10, 1))],   
            [sg.Text('p: ')],   
			      [sg.Radio('5', 'p', size=(10, 1)), sg.Radio('10', 'p', size=(10, 1)), sg.Radio('15', 'p', size=(10, 1))],     
            [sg.Submit('Calcular!'), sg.Quit('Sair')]]      
	window2 = sg.Window('Machine Learning', layout)
	event, values2 = window2.Read()  
	if event == 'Sair' or event is None:
		window2.Close()
		return()
	
# values é um vetor que corresponde aos botões Radio
# a correspondência é: values -> [100, 1000, 4000, 5, 10, 15]	
	if values2[0]: ndig_treino = 100
	elif values2[1]: ndig_treino = 1000
	elif values2[2]: ndig_treino = 4000		
	if values2[3]: p = 5
	elif values2[4]: p = 10
	elif values2[5]: p = 15

# execução das tarefas
# verifica se foram selecionados de fato os valores de ndig_treino e p	
	if p + ndig_treino > 104:
		window2.Close()  
		print("Tarefa principal: Machine Learning")
		print("ndig_treino: " + str(ndig_treino))
		print("p: " + str(p))
		print()
		machine_learning(ndig_treino, p)	
	else: 
		sg.PopupOK('Erro')
		window2.Close()
		return()
		
# imprime apenas 3 casas decimais das matrizes
np.set_printoptions(formatter={'float': lambda x: "{0:0.3f}".format(x)})

if __name__ == '__main__':
	while True:
		event, value = window.Read()      
		if event == 'Rot-Givens': teste_rot_givens()
		elif event == 'A': teste_a() 
		elif event == 'B': teste_b()  
		elif event == 'C': teste_c()  
		elif event == 'D': teste_d() 
		elif event == 'A = WH': tarefa_2()   		   		
		elif event == 'Machine Learning': machine()
		elif event == 'Sair' or event == sg.WINDOW_CLOSED or event is None:  
			break