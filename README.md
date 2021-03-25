# PyTorch Geometric Example

Developed to run on 0.14.2 (hotpatches needed)

## Build dockerfile

```sh
docker build . -t ptgm
```

## Launch the cluster

```sh
det-deploy local cluster-up --no-gpu
```

## Run the experiment locally

```sh
det e create const.yaml . --local --test
```

## Run the experiment on the cluster

```sh
det e create const.yaml . -f
```
