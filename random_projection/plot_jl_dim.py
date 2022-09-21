# -*- coding: utf-8 -*-

from sklearn.random_projection import johnson_lindenstrauss_min_dim
import matplotlib.pyplot as plt

eps_arr = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
n_samples_arr = [100, 300, 500, 1000, 3000, 5000, 7500, 10000]

eps_fixed = 0.1
n_samples_fixed = 500

jl_dims_samples = []
jl_dims_eps = []

for n_samples in n_samples_arr:
    jl_dims_samples.append(johnson_lindenstrauss_min_dim(n_samples, eps=eps_fixed))


for eps in eps_arr:
    pass
    jl_dims_eps.append(johnson_lindenstrauss_min_dim(n_samples_fixed, eps=eps))

# else epsilon looks like varepsilon
plt.rcParams["mathtext.fontset"] = "cm"
plt.rc('axes', labelsize=14)
plt.rc('legend', fontsize=14)

#plt.yscale('log')
plt.plot(eps_arr, jl_dims_eps, label='$w_+ = 500$', color='orange')
plt.xlabel('$\epsilon$')
plt.ylabel('# of dimensions $k$')
plt.legend()
plt.show()
plt.savefig('plt/rp_jl_eps.eps')

#plt.yscale('log')
plt.figure()
plt.plot(n_samples_arr, jl_dims_samples, label='$\epsilon=0.1$', color='orange')
plt.xlabel('$w_+$')
plt.ylabel('# of dimensions $k$')
plt.legend()
plt.show()
plt.savefig('plt/rp_jl_w.eps')