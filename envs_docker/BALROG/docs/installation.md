# Installation

We advise using conda for the installation
```
conda create -n balrog python=3.10 -y
conda activate balrog

git clone https://github.com/balrog-ai/BALROG.git
cd BALROG
pip install -e .
balrog-post-install
```


Test your installation by running:

```
pytest balrog/tests
```