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
docker run -it --rm parallelworks/pflotran /bin/bash
```
If you don't already have the container downloaded to your computer, Docker will automatically pull the container from a public registry.

