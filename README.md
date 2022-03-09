      _____  _____  _____ __  __ ____                _______            
     |  __ \|_   _|/ ____|  \/  |  _ \              |__   __|           
     | |__) | | | | (___ | \  / | |_) | __ _ _ __ _   _| |_ __ ___  ___ 
     |  _  /  | |  \___ \| |\/| |  _ < / _` | '__| | | | | '__/ _ \/ _ \
     | | \ \ _| |_ ____) | |  | | |_) | (_| | |  | |_| | | | |  __/  __/
     |_|  \_\_____|_____/|_|  |_|____/ \__,_|_|   \__, |_|_|  \___|\___|
                                                   __/ |                
                                                  |___/                 

RISMBaryTree
============

   A work-in-progress library for GPU computation of N-body interactions for RISM long-range
   kernels, using barycentric Lagrange polynomial interpolation treecodes.
   This code employs CUDA for its GPU implementation.
   For the MPI+OpenACC version of general purpose BaryTree,
   [see this repository](https://github.com/Treecodes/BaryTree).


   Authors:  
   - Leighton W. Wilson
   - Nathan J. Vaughn
   - Erick Aitchison
   - Tyler Luchko
   - Ray Luo
   

Building
--------
This project uses CMake to manage and configure its build system. In principle, 
building this project is as simple as executing the following from the top level
directory of BaryTree:

```bash
mkdir build; cd build; export CC=<C compiler> export CUDACXX=<CUDA compiler>; cmake ..; make
```

You may also need to specify `CMAKE_CUDA_FLAGS` when you configure with `cmake`.
For example: `cmake .. -DCMAKE_CUDA_FLAGS="-gencode arch=compute_75,code=sm_75"`.
For more information on building and installing, see __INSTALL.md__ in this directory.


References
----------
   Please refer to the following references for more background:
        
   - N. Vaughn, L. Wilson, and R. Krasny, A GPU-accelerated barycentric 
            Lagrange treecode, submitted to _Proc. 21st IEEE Int.
	    Workshop Parallel Distrib. Sci. Eng. Comput._ (PDSEC 2020) 
	    (2020).
	    
   - R. Krasny and L. Wang, A treecode based on barycentric Hermite 
            interpolation for electrostatic particle interactions,
	    _Comput. Math. Biophys._ __7__ (2019), 73-84.
		
   - H. A. Boateng and R. Krasny, Comparison of treecodes for
            computing electrostatic potentials in charged particle 
	    systems with disjoint targets and sources,
            _J. Comput. Chem._ __34__ (2013), 2159-2167.	
	   
   - J.-P. Berrut and L. N. Trefethen, Barycentric Lagrange interpolation,
            _SIAM Rev._ __46__ (2004), 501-517.

   - Z.-H. Duan and R. Krasny, An adaptive treecode for computing
            nonbonded potential energy in classical molecular systems,
            _J. Comput. Chem._ __22__ (2001), 184–195.

                                                    
License
-------
Copyright © 2019-2022, The Regents of the University of Michigan. Released under the [MIT License](LICENSE).
