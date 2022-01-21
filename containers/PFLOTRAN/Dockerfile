# Use official Ubuntu 20.04 as base image

FROM ubuntu:20.04

# Set the working directory to /app

WORKDIR /app
ADD ./test.sh .

# Note that RUN will add a layer,
# CMD just executes a command (actually, CMD
# is meant to set a DEFAULT command, not
# really to spit out test results), and
# ENV sets an environment variable (and
# also writes a layer in the process).
# We want to minmize the numbers of
# layers added for a variety of reasons
# (performance and stability), see:
# https://docs.docker.com/develop/develop-images/dockerfile_best-practices/
# so please comment out CMD lines for
# production images pushed back to DockerHub.
# Key things to note here:
# 1) Never RUN apt-get upgrade
# 2) Always RUN apt-get upgrade and install together
CMD ["pwd"]
CMD ["ls" "-l"]
CMD ["./test.sh test line in Dockerfile"]

# Valgrind is encouraged by PETSC, but I think
# it might be an optional memory leak tracking
# tool. Cmake is required by ParMETIS
RUN apt-get update && apt-get install -y \
    apt-utils \
    build-essential \
    cmake \
    gcc \
    gfortran \
    g++ \
    git-core \
    python \
    valgrind \
    wget

# Check some versions
#CMD ["gcc", "--version"]
#CMD ["g++", "--version"]
#CMD ["gfortran", "--version"]

# PETSC, need to use current suppored version.
# As listed on the PFLOTRAN website - note this
# is not the same as the current version
# (3.11.2, as of June 3, 2019).
#
# PETSC can be configured to download and install
# its own dependencies, in particular HDF5 which
# is usually a bear to compile.  This config line
# will enable MPI, unstructured grids, and grab
# additional dependences providing full functionality
# for most users.  Reduced/optimized functionality
# can be tested later.  The steps executed here
# really do grab the additional source code and
# config/compile it from scratch, so this takes
# a good while.
#
# PETSC configuration requires python (2.7, not 3).
# NEED TO VERIFY THESE DEFAULT ENVIRONMENT VARIABLES
ENV PETSC_DIR /app/petsc
ENV PETSC_ARCH arch-linux2-c-opt
RUN git clone https://bitbucket.org/petsc/petsc petsc; \
    cd petsc; \
    git checkout v3.10.2; \
    ./configure --CFLAGS='-O3' --CXXFLAGS='-O3' --FFLAGS='-O3' --with-debugging=no --download-mpich=yes --download-hdf5=yes --download-fblaslapack=yes --download-metis=yes --download-parmetis=yes; \
    make all;
    cd ../

# PFLOTRAN
RUN git clone https://bitbucket.org/pflotran/pflotran; \
    cd pflotran/src/pflotran; \
    make pflotran

# Left over from larval-parameter-sweep, don't
# know if it is needed here or not.
#RUN apt-get install -y software-properties-common

CMD ["echo", "Hello from Docker!"]

# If we were to try multi-stage builds, then
# at this point, one would create a new FROM
# command and try to transfer PETSC_ARCH and
# PFLOTRAN.  But I don't know whether the
# other system libs are statically linked or not.