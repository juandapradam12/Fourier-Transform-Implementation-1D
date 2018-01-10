import numpy as np 

## Lectura de Datos 

Datos=np.loadtxt('funcion.dat')

x = Datos[:,0] 
y = Datos[:,1] 

## Constantes 

N = np.shape(Datos)[0]

## Transformada de Fourier

def T_Fourier(z): 
	n = np.arange(N)
	k = np.reshape(n, (N, 1))
	M_nk = np.exp(-(2j*np.pi*n*k)/N)
	T_z = np.dot(M_nk, z)
	return T_z
