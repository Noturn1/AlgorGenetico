import numpy as np
import matplotlib
import matplotlib.pyplot as plt
matplotlib.use('TkAgg') 

# Define função de otimização
def funcao_objetivo(x, y):
    return x**2 + y**2 + (3*x + 4*y - 26)**2
 
# Codificação e decodificação binária
def binario_para_decimal(binario, limite_inferior, limite_superior, precisao=3):
    valor_inteiro = int(binario[1:], 2)  # Para ignorar o bit de sinal
    sinal = -1 if binario[0] == '1' else 1
    decimal = sinal * (valor_inteiro / (2**(len(binario) - 1) - 1)) * (limite_superior - limite_inferior) + limite_inferior
    return round(decimal, precisao)

def decimal_para_binario(valor, bits=16):
    limite_inferior, limite_superior = 0, 20
    sinal = '1' if valor < 0 else '0'
    valor_normalizado = abs(valor - limite_inferior) / (limite_superior - limite_inferior)
    valor_inteiro = int(valor_normalizado * (2**(bits - 1) - 1))
    binario = bin(valor_inteiro)[2:].zfill(bits - 1)
    return sinal + binario

# Gerar a população
def gerar_populacao(tamanho_populacao, bits):
    return [''.join(np.random.choice(['0', '1'], size=bits)) for _ in range(tamanho_populacao)]

# Avaliar o fitness
def avaliar_populacao(populacao, bits):
    return [funcao_objetivo(binario_para_decimal(indiv[:bits//2], 0, 10),
                            binario_para_decimal(indiv[bits//2:], 0, 20)) for indiv in populacao]

# Seleção por ranking
def selecao_ranking(populacao, fitness):
    ranking = np.argsort(fitness)
    probabilidades = np.arange(len(fitness), 0, -1) / sum(range(1, len(fitness) + 1))
    selecionados = np.random.choice(populacao, size=len(populacao), p=probabilidades[ranking])
    return selecionados

# Cruzamento de dois pontos
def cruzamento(individuo1, individuo2):
    ponto1, ponto2 = sorted(np.random.choice(range(1, len(individuo1) - 1), size=2, replace=False))
    filho1 = individuo1[:ponto1] + individuo2[ponto1:ponto2] + individuo1[ponto2:]
    filho2 = individuo2[:ponto1] + individuo1[ponto1:ponto2] + individuo2[ponto2:]
    return filho1, filho2

# Mutação por inversão binária
def mutacao(individuo, taxa_mutacao):
    individuo_mutado = list(individuo)
    for i in range(len(individuo)):
        if np.random.rand() < taxa_mutacao:
            individuo_mutado[i] = '1' if individuo_mutado[i] == '0' else '0'
    return ''.join(individuo_mutado)

# Algoritmo Genético
def algoritmo_genetico(tamanho_populacao, num_geracoes, taxa_cruzamento, taxa_mutacao, bits=32):
    populacao = gerar_populacao(tamanho_populacao, bits)
    fitness_medio_por_geracao = []

    for geracao in range(num_geracoes):
        fitness = avaliar_populacao(populacao, bits)
        fitness_medio_por_geracao.append(np.mean(fitness))

        # Elitismo
        elite_index = np.argmin(fitness)  # Seleciona o índice do melhor indivíduo
        elite = populacao[elite_index]

        # Seleção e cruzamento
        selecionados = selecao_ranking(populacao, fitness)
        nova_populacao = []
        for i in range(0, tamanho_populacao, 2):
            if np.random.rand() < taxa_cruzamento:
                filho1, filho2 = cruzamento(selecionados[i], selecionados[i+1])
                nova_populacao.extend([filho1, filho2])
            else:
                nova_populacao.extend([selecionados[i], selecionados[i+1]])

        # Mutação
        populacao = [mutacao(individuo, taxa_mutacao) for individuo in nova_populacao]

        # Substituir o pior indivíduo pela elite
        populacao[np.argmax(fitness)] = elite
    
        # Gráfico da geração atual
        plt.figure(figsize=(6,4))
        x_vals = [binario_para_decimal(str(ind[:bits//2]), 0, 10) for ind in populacao]
        y_vals = [binario_para_decimal(str(ind[bits//2:]), 0, 20) for ind in populacao]
        plt.scatter(x_vals, y_vals, c='blue', label='Indivíduos')
        plt.xlabel('x')
        plt.ylabel('y')
        plt.title(f'Geração {geracao + 1}')
        plt.legend()
        plt.show(block=False)
        plt.pause(1)
        plt.close()

    # Gráfico do fitness médio
    plt.figure(figsize=(8, 5))
    plt.plot(range(num_geracoes), fitness_medio_por_geracao, marker='o')
    plt.xlabel('Geração')
    plt.ylabel('Fitness Médio')
    plt.title('Convergência do Algoritmo Genético')
    plt.grid()
    plt.show()
    plt.close()

# Parâmetros do algoritmo
algoritmo_genetico(
    tamanho_populacao=50,
    num_geracoes=300,
    taxa_cruzamento=0.8,
    taxa_mutacao=0.0075,
)
            

