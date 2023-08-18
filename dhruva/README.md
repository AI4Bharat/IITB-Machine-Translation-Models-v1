```
docker build -t iitb .
```

```
docker run --shm-size=256m --gpus=1 --rm -v ${PWD}/:/models/dhruva -p 8000:8000 -p 8001:8001 -p 8002:8002 iitb
```
