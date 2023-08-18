```
cd dhruva
```

```
export RESOURCE_GROUP=Dhruva-prod
export WORKSPACE_NAME=dhruva--central-india
export DOCKER_REGISTRY=dhruvaprod
```

```
az acr login --name $DOCKER_REGISTRY
docker tag iitb $DOCKER_REGISTRY.azurecr.io/nmt/triton-nmt-monolingual:latest
docker push $DOCKER_REGISTRY.azurecr.io/nmt/triton-nmt-monolingual:latest
```

```
az ml environment create -f azure_ml/environment.yml -g $RESOURCE_GROUP -w $WORKSPACE_NAME
```

```
az ml model create --file azure_ml/model.yml --resource-group $RESOURCE_GROUP --workspace-name $WORKSPACE_NAME
```

```
az ml online-endpoint create -f azure_ml/endpoint.yml -g $RESOURCE_GROUP -w $WORKSPACE_NAME
```

```
az ml online-deployment create -f azure_ml/deployment.yml --all-traffic -g $RESOURCE_GROUP -w $WORKSPACE_NAME
```
