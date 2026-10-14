import numpy as np
import matplotlib.pyplot as plt


class Perceptron():

    def __init__(self, forma, pesos=None, bias=None):
        """
        forma : tuple
        Una tupla que contiene el numero de activaciones en cada capa de la red.
        ej.
        (1, 3, 3, 1)
        Corresponde a una red de una 1 entrada => 3 activaciones => 3 activaciones => 1 salida.

        pesos : list(np.ndarray)
        Una lista que contiene los pesos en cada capa de activaciones (en la primera capa se inserta None).
        Si no se especifican, se escogen aleatoriamente.
        Cada array de pesos debe tener shape (1, a, b) representando el peso que conecta la activacion 'a' en la capa n
        con la activacion 'b' en la capa n - 1.

        bias : list(np.ndarray)
        Una lista que contiene los bias en cada capa de activaciones (en la primera capa se inserta None).
        Si no se especifican, se escogen aleatoriamente.
        """
        self.forma = forma
        self.profunidad = len(forma)

        # Si no se especifican pesos, generarlos aleatoriamente
        if not pesos:
            self.init_pesos()
        else:
            self.pesos = pesos

        # Si no se especifican bias, generarlos aleatoraimente
        if not bias:
            self.init_bias()
        else:
            self.bias = bias
        
        
    def init_pesos(self):
        self.pesos = [None]
        for a_antes, a in zip(self.forma[:-1], self.forma[1:]):
            w = np.random.rand(1, a, a_antes)
            self.pesos.append(w)

    
    def init_bias(self):
        self.bias = [None]
        for i in self.forma[1:]:
            b = np.zeros((1, i, 1), dtype=np.float64)
            self.bias.append(b)


    def func_act(self, x):
        """
        La funcion de activacion de cada neurona de la red
        """
        return np.maximum(0, x)
    
    
    def deriv_act(self, x):
        """
        La primera derivada de la funcion de activacion
        """
        return (x > 0).astype(np.float64)


    def propagar_adelante(self, entrada):
        """
        Evalua un numero arbitrario de entradas en la red neruonal.

        parametros --------
        entrada : np.ndarray (i, a, 1).
        Entrada para la red neuronal, la dimension 'i' representa la entrada i-esima y 'a' representa la componente
        a-esima del vector de entradas.

        devuelve ---------
        pre_activaciones : list
        Una lista que contiene los valores de las preactivaciones en las capas de activaciones y salida.

        activaciones : list
        Una lista que contiene los valores de activaciones en las capas de activaciones y salidas
        """
        activaciones = [entrada]
        pre_activaciones = [None] # No hay preactivaciones en la primera capa

        for w, b in zip(self.pesos[1:], self.bias[1:]):
            act_anterior = activaciones[-1]
            z = w @ act_anterior + b
            act = self.func_act(z)

            activaciones.append(act)
            pre_activaciones.append(z)
        
        return activaciones, pre_activaciones
    

    def propagar_atras(self, activaciones, pre_activaciones, X, Y):
        """
        Dado un conjunto de evaluaciones al perceptron que dejo un estado de activaciones y pre_activaciones.
        Esta funcion calcula el gradiente de la funcion costo (diferencias al cuadrado) respecto a los pesos y biases.

        parametros -----------
        activaciones : list
        Lista que contiene las activaciones del perceptron en cada capa

        pre_activaciones : list
        Lista que contiene los pesos del perceptron en cada capa

        X : np.ndarray (i, a, 1)
        Entradas de los datos de entrenamiento

        Y : np.ndarray (i, a, 1)
        Entradas de los datos de entrenamiento

        devuelve -----------------
        returns grad_peso, grad_bias

        grad_peso : list
        Lista que contiene las derivadas parciales de la funcion costo respecto a los pesos en cada capa

        grad_bias : list
        Lista que contiene las derivadas parciales de la funcion costo respecto a los bias en cada capa
        """
        num_datos = X.shape[0]
        grad_peso = [None for _ in range(self.profunidad)]
        grad_bias = [None for _ in range(self.profunidad)]

        # Obtenemos activaciones y preactivaciones de la ultima capa (capa de salida)
        act_l = activaciones[-1]
        pre_act_l = pre_activaciones[-1]

        # Calculamos el bridge de la ultima capa
        # bridge es el conjunto de las siguientes derivadas parciales: \frac{\partial C}{\partial \mathrm{Z}_{a}^{li}}
        # Los indices simbolizan: dato experimantal i-esimo, en la capa l-esima de la red y la neruona a-esima 
        bridge = 2 * (act_l - Y) * self.deriv_act(pre_act_l) / num_datos

        # Propagamos hacia atras
        for n in range(1, self.profunidad):
            capa = self.profunidad - n

            # Obtenemos activaciones, preactivaciones y pesos de la capa actual y la anterior
            w = self.pesos[capa]
            act_antes = activaciones[capa - 1]
            preact_antes = pre_activaciones[capa - 1]

            # Transponemos activaciones de la capa anterior
            act_antes = np.transpose(act_antes, (0, 2, 1)) # (i, b, 1) => (i, 1, b)

            # Derivada de la funcion costo respecto a pesos y bias
            dc_dw = np.sum(bridge * act_antes, axis=0, keepdims=True)
            dc_db = np.sum(bridge, axis=0, keepdims=True)

            # Añadimos derivadas parciales a listas grad_peso y grad_bias
            grad_peso[capa] = dc_dw
            grad_bias[capa] = dc_db

            # En la penultima capa, se debe omitir el calculo de bridge
            if capa != 1:
                # Transponemos preactivaciones
                preact_antes = np.transpose(preact_antes, (0, 2, 1)) # (i, b, 1) => (i, 1, b)

                # Actualizamos bridge
                bridge = self.deriv_act(preact_antes) * np.sum(bridge * w, axis=1, keepdims=True)

                # Transponemos bridge
                bridge = np.transpose(bridge, (0, 2, 1)) # (i, 1, b) => (i, b, 1)


        return grad_peso, grad_bias

            
    def entrenar(self, X, Y, pasos=100, mu=0.2, progreso=False):
        """
        Entrena una red usando propagacion hacia atras para el set {entradas, salidas}.

        parametros -----------
        red : function
        La red neuronal, debe tomar como argumentos (pesos, bias, entrada) y devolver (activaciones, pre_activaciones)

        X : np.ndarray (i, a, 1)
        Las entradas para los datos de entrenamiento, los ejes simbolizan: i : ejemplo i-esimo. a : componente a-esima del
        vector de entradas

        Y : np.ndarray (i, a, 1)
        Las salidas que debe producir la red correspondientes a las entradas de entrenamiento, los ejes simbolizan:
        i : ejemplo i-esimo. a : componente a-esima del vector de salidas
        """
        if pasos > 100:
            tope = int(pasos / 100)
        else:
            tope = pasos

        prog = 0
        contador = 0
        for _ in range(pasos):

            # Progreso del entrenamiento de la red
            contador += 1
            if contador == tope:
                prog += 1
                print(f'Progreso {round(prog, 1)} %')
                contador = 0

            # Foward pass
            activaciones, pre_activaciones = self.propagar_adelante(X)

            # Back pass
            grad_w, grad_b = self.propagar_atras(activaciones, pre_activaciones, X, Y)
            
            # Descenso del gradiente
            for i in range(1, self.profunidad):
                # Obtenemos pesos, bias y derivadas en cada capa del perceptron
                dc_dw = grad_w[i]
                dc_db = grad_b[i]
                w = self.pesos[i]
                b = self.bias[i]

                # Calculamos nuevos pesos y bias
                w = w - mu * dc_dw
                b = b - mu * dc_db

                # Actualizamos pesos y bias del perceptron
                self.pesos[i] = w
                self.bias[i] = b

    
    def evaluar(self, entrada):
        """
        """
        act, preact = self.propagar_adelante(entrada)
        salida = act[-1]
        return salida
    



    

    

pepe = Perceptron((1, 10, 1))
xdata = np.linspace(0, 6, 100)[:, None, None]
ydata = np.exp(xdata)
pepe.entrenar(xdata, ydata, pasos=30000, mu=0.01)

x = np.linspace(0, 6, 500)
x = x[:, None, None]
y = pepe.evaluar(x)

plt.style.use('ggplot')
fig, ax = plt.subplots()
ax.scatter(xdata[:, 0, 0], ydata[:, 0, 0], color='black')
ax.plot(x[:, 0, 0], y[:, 0, 0], color='red')
plt.show()
