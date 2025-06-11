**Status:** Maintenance (expect bug fixes and minor updates)

Welcome to Spinning Up in Deep RL! 
==================================

This is an educational resource produced by OpenAI that makes it easier to learn about deep reinforcement learning (deep RL).

For the unfamiliar: [reinforcement learning](https://en.wikipedia.org/wiki/Reinforcement_learning) (RL) is a machine learning approach for teaching agents how to solve tasks by trial and error. Deep RL refers to the combination of RL with [deep learning](http://ufldl.stanford.edu/tutorial/).

This module contains a variety of helpful resources, including:

- a short [introduction](https://spinningup.openai.com/en/latest/spinningup/rl_intro.html) to RL terminology, kinds of algorithms, and basic theory,
- an [essay](https://spinningup.openai.com/en/latest/spinningup/spinningup.html) about how to grow into an RL research role,
- a [curated list](https://spinningup.openai.com/en/latest/spinningup/keypapers.html) of important papers organized by topic,
- a well-documented [code repo](https://github.com/openai/spinningup) of short, standalone implementations of key algorithms,
- and a few [exercises](https://spinningup.openai.com/en/latest/spinningup/exercises.html) to serve as warm-ups.

Get started at [spinningup.openai.com](https://spinningup.openai.com)!

## Install

**spinningup**
```shell
# update pip
conda create -n spinningup_py36 python=3.6
conda activate spinningup_py36
sudo apt-get update && sudo apt-get install libopenmpi-dev
cd spinningup

pip install --upgrade pip
pip install -e .

# install opencv
conda install -c conda-forge opencv
pip install opencv-python==4.5.1.48
pip install -e .
```


**install mujoco_py**
```shell
wget https://github.com/google-deepmind/mujoco/releases/download/3.3.2/mujoco-3.3.2-linux-x86_64.tar.gz
tar -xzvf mujoco-3.3.2-linux-x86_64.tar.gz

vim ~/.bashrc
# export MUJOCO_PY_MUJOCO_PATH=/home/sukui/01.software/mujoco210
# export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/home/sukui/01.software/mujoco210/bin

source ~/.basrch

cd ~/01.software/mujoco210/bin/
./simulate ../model/humanoid.xml

git clone https://github.com/openai/mujoco-py.git
cd mujoco-py
pip3 install -U 'mujoco-py<2.2,>=2.1'
pip3 install -r requirements.txt
pip3 install -r requirements.dev.txt
pip install "cython<3"
python3 setup.py install


# test
cd spinningup
python -m spinup.run ppo --env Walker2d-v2 --exp_name walker
```
Citing Spinning Up
------------------

If you reference or use Spinning Up in your research, please cite:

```
@article{SpinningUp2018,
    author = {Achiam, Joshua},
    title = {{Spinning Up in Deep Reinforcement Learning}},
    year = {2018}
}
```