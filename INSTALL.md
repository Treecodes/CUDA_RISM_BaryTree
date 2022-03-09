Installing
----------

In a best case scenario, building and installing the libraries and examples should be as simple as this:

```bash
mkdir build; cd build; export CC=<C compiler>; export CUDACXX=<CUDA compiler>;
cmake .. -DCMAKE_INSTALL_PREFIX=<install location>;
make -j install;
```

This assumes that you have a few things:
1. a sane C compiler,
2. a sane CUDA compiler,
3. CMake version 3.9 or newer.

CMake Flags
-----------
The most useful CMake flags to use during configure are listed below. When passing a flag
to `cmake` during configure, recall that it takes the form `-D<flag>=value`.
| Flag                   | Option/ Value                | Description
|------------------------|------------------------------|------------
| `CMAKE_RELEASE_TYPE`   | Debug, Release               | Build either the Debug or Release version.
| `ENABLE_CUDA_BUILD`    | ON, OFF                      | Toggle whether to build the GPU versions.
| `CMAKE_INSTALL_PREFIX` | `<where to install>`         | Specify install location for `make install`.
| `BUILD_EXAMPLES`       | ON, OFF                      | Toggle whether to build examples.
| `BUILD_SHARED_LIBS`    | ON, OFF                      | Toggle whether to build libraries as shared.
| `CMAKE_CUDA_FLAGS`     | `options for CUDA`           | Used to specify target compute capability.
