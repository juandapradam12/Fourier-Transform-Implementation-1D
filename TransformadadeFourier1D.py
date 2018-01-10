import numpy as np 

def T_Fourier(z): # Entra por parametro un array Nx1 (256x1)
	# Indices del Kernel
	n = np.arange(N)
	k = np.reshape(n, (N, 1))
	# Matriz del Kernel de la transformacion NxN (256x256)
	M_nk = np.exp(-(2j*np.pi*n*k)/N)
	# Datos en el espacio de frecuencias Nx1 (256x1)
	T_z = np.dot(M_nk, z)
	return T_z
