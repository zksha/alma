# Building Images Locally

To build and run an image (e.g. `Dockerfile`) do:

```bash
docker build --file docker/Dockerfile . --tag balrog
docker run -it --gpus all --rm --name balrog balrog
# or alternatively if you don't have GPUs
docker run -it --name balrog balrog
```
