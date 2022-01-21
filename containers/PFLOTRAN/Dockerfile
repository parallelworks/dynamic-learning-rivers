# Use official Ubuntu 20.04 as base image

FROM ubuntu:20.04

# Set the working directory to /app

WORKDIR /app

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
# so lines are often added together with ;
# (end of command markers) and \ (line
# continuation markers.)
#
# Key things to note here:
# 1) Never RUN apt-get upgrade
# 2) Always RUN apt-get upgrade and install together

#------------------------------------
# Configure time zone so apt-get update
# does not hang.  Thanks to Grigor Khachatryan at:
# https://grigorkh.medium.com/fix-tzdata-hangs-docker-image-build-cdb52cc3360d

ENV TZ=America/New_York
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone
#------------------------------------

# Valgrind is encouraged by PETSC, but I think
# it might be an optional memory leak tracking
# tool. Cmake is required by ParMETIS.  PETSC
# config requires python (not just python3).
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

# PETSC, need to use current suppored version.
# Container build is interrupted if the correct
# PETSC version is not included and it says which
# version to use (e.g. v3.16.2), which is included
# below.
#
# PETSC can be configured to download and install
# its own dependencies, in particular HDF5 which
# is usually a bear to compile.  This config line
# will enable MPI, unstructured grids, and grab
# additional dependences providing full functionality
# for most users, including zlib compression with
# HDF5.  Reduced/optimized functionality
# can be tested later.  The steps executed here
# really do grab the additional source code and
# config/compile it from scratch, so this takes
# a good while.
#
# PETSC configuration requires python and python3.
# Cannot find hd5.mod? with v3.16.3, but b/c missing --download-hdf5-fortran-bindings=yes

ENV PETSC_DIR /app/petsc
ENV PETSC_ARCH arch-linux-c-opt
RUN git clone https://gitlab.com/petsc/petsc.git; \
    cd petsc; \
    git checkout v3.16.3; \
    ./configure --CFLAGS='-O3' --CXXFLAGS='-O3' --FFLAGS='-O3' --with-debugging=no --download-mpich=yes --download-hdf5=yes --download-hdf5-fortran-bindings=yes --download-hdf5-configure-arguments="--with-zlib=yes" --download-fblaslapack=yes --download-metis=yes --download-parmetis=yes; \
    make all; \
    cd ../

# PFLOTRAN
RUN git clone https://bitbucket.org/pflotran/pflotran; \
    cd pflotran/src/pflotran; \
    git checkout maint/v4.0; \
    make pflotran

# If we were to try multi-stage builds, then
# at this point, one would create a new FROM
# command and try to transfer PETSC_ARCH and
# PFLOTRAN.  But I don't know whether the
# other system libs are statically linked or not.