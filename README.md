# simulator_bus

## Setup

### **Step 1** -- Clone Repo

* `git clone git@github.com:bonny2016/simulator_bus.git`

* `cd simulator_bus`

### **Step 2** -- Create And Activate Conda Environment

* Note: You can download the miniconda installer from
https://conda.io/miniconda.html. OR, you may use any other Python environment with Python 3.10.13

File conda_env.yml containes all the packages that will be needed.

* `conda env create -f conda_env.yml`

* `conda activate pytorch_env`

## Run simulator
```
  python -m simulator.environ
```

## Train Actor-Critic Model:
```
  python main.py
```
