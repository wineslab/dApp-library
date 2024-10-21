# Install libiq guides

## prerequisites

Swig

```
apt install swig
```

Export the path

`export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/lib`

## libiq 

- clone the repo in a recursive way
- `git submodule update --init --recursive`

```
Per far funzionare tutto bisogna fare 
1 in prova.py bisogna mettere prima di ogni cosa:
	`import sys`
	`sys.path.append('/home/user/Desktop/project/libiq-101')`
2 sostituire tutti i path con "home/user/Desktop/project/libiq-101" invece di "/root/libiq-101"
```

##############################################################################

1)installare matio
	1.1)installare le dipendenze
		1.1.1)installare zlib
			1.1.1.1)andare nella cartella libiq-101/libs/zlib/
			1.1.1.2)scrivere sul terminale `cmake .`
			1.1.1.3)scrivere sul terminale `cmake --build .`
			1.1.1.4)scrivere sul terminale `cmake --install .`
			1.1.1.5)in questo modo zlib viene messo in /usr/local/include/
		1.1.2)installare HDF5
			1.1.2.1)download source code from https://github.com/HDFGroup/hdf5/archive/refs/tags/hdf5-1_14_3.tar.gz
			1.1.2.2)unzip folder with tar -xzf
			1.1.2.3)enter folder then type `./configure --prefix=/usr/local/include/hdf5 --enable-cxx`
			1.1.2.4)type `make`
			1.1.2.5)type `make check`
			1.1.2.6)type `make install`
			1.1.2.7)type `make check-install`
			1.1.2.8) in questo modo verrà installato in /usr/local/include/hdf5/
	1.2)build matio
		1.2.1)apt-get install libtool
		1.2.2)./autogen.sh
		1.2.3)./configure --enable-mat73=yes --with-default-file-ver=7.3 --with-hdf5=/usr/local/include/hdf5/
		1.2.4)make
		1.2.5)make install PREFIX=/usr/local/include/matio
		1.2.6)in questo modo verranno aggiunti i file .h in /usr/local/include/

##############################################################################

1)installare libsigmf
	1.1)entrare in libiq-101/libs/libsigmf/ e scrivere `mkdir build` e fare `cd build`
	1.2)scrivere `cmake ../`
	1.3)scrivere `make -j6`
	1.4)scrivere `make install`

##############################################################################

1)installare fftw
	1.1)scaricare pacchetto dalla pagina https://fftw.org/fftw-3.3.10.tar.gz
	1.2)fare `tar -xzf pacchetto.tar.gz`
	1.3)entrare nella cartella
	1.4)`./configure --enable-shared --with-pic`
	1.5)`make -j6`
	1.6)`make install`
	1.7)così i file sarano messi in /usr/local/include