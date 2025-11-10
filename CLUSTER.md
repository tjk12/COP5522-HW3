On HPC clusters you may need to load an MPI module before building/running.

Typical steps:

```sh
module avail
module load openmpi   # or the cluster-specific MPI module/version
make clean && make
```

If `make` errors with "No MPI compiler wrapper found", load the MPI module or set the `CXX` environment variable to your MPI C++ wrapper, e.g.

```sh
export CXX=mpic++
```
