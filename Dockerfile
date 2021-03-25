FROM determinedai/environments:py-3.6.9-pytorch-1.7-tf-1.15-cpu-0.9.0

RUN pip install torch-scatter -f https://pytorch-geometric.com/whl/torch-1.7.0+cpu.html && \
    pip install torch-sparse -f https://pytorch-geometric.com/whl/torch-1.7.0+cpu.html && \
    pip install torch-cluster -f https://pytorch-geometric.com/whl/torch-1.7.0+cpu.html && \
    pip install torch-spline-conv -f https://pytorch-geometric.com/whl/torch-1.7.0+cpu.html && \
    pip install torch-geometric
