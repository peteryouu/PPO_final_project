# Evaluating Proximity Policy Optimization Algorithms on Atari Environments

The original paper for implementation is found here: https://arxiv.org/abs/1707.06347

## Environment

The code was initially written in Jupyter Notebooks, and converted into Python files. The training was done over a Ryzen 5900x CPU and over Python 3.8 

## Training PPO on Atari

1. Clone or download this repo

2. Unzip the downloaded folder

3. Go to root directory of project and open up Command Prompt in the directory and run:

4. pip install -r requirements.txt

5. Unzip the ROMs.zip at the same root directory

6. Run python -m atari_py.import_roms .\ROMS\ROMS in command line

7. Run the code via python train.py

## References
1. https://blog.varunajayasiri.com/ml/ppo.html
2. https://towardsdatascience.com/optimized-deep-q-learning-for-automated-atari-space-invaders-an-implementation-in-tensorflow-2-0-80352c744fdc
3. https://arxiv.org/abs/1707.06347
4. https://github.com/deepanshut041/Reinforcement-Learning/blob/master/cgames/01_ping_pong/
