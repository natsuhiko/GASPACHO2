# GASPACHO2

GASPACHO is a Gaussian process (GP) based approach to extract latent factors from single-cell data and to map dynamic genetic associations along the latent factors.

The software is now implemented on GPGPU with CUDA C (ver 11) to handle more than 100K cells.

# Basic installation

Before you make the code, you need to install GSL (https://www.gnu.org/software/gsl/). The code is tested with GSL-2.8.

You need to modify CFLAGS, NVCCFLAGS and LDFLAGS to make.

	# login to a server with NVIDIA h100 graphic cards 
	make -f Makefile_h100


