# Instructions for building and using this PFLOTRAN container

## Contents

This directory has the `Dockerfile` which gives Docker the instructions
to build the container.

## Build container

This step is only needed if you want to make significant updates to the container because this container has already been built and it is a public image (see `Use container` below).

To build the container, while in this directory,
type the following into the command line:
```bash
docker build -t parallelworks/pflotran .
```
**Notes:**
1. Depending on your Docker setup, you may need to change `docker` to
`sudo docker`.
2. Don't forget the `.` at the end of the line.  This tells Docker where the build context is (the files to include in the container are all the files in this directory).

## Use container

You can start the container in a `bash` shell with:
```bash
docker run -it --rm parallelworks/pflotran:v4.0 /bin/bash
```
If you don't already have the container downloaded to your computer, Docker will automatically pull the container from a public registry.  The last part of the container name (the "tag" or `v4.0` was added later, see below.  If it is not specified, Docker will by default pull the most recent version.)

## Push container

Once the container is build, I tagged it with the PFLOTRAN
version number (v4.0) with:
```bash
docker tag parallelworks/pflotran parallelworks/pflotran:v4.0
```
and then I pushed it into the public DockerHub registry
under the parallelworks repository:
```bash
docker push parallelworks/pfortran:v4.0
```
Now, with
```bash
docker pull parallelworks/pflotran:v4.0
```
this container can be downloaded to any machine
with Docker.  The download is about 3GB.
