Install the libraries in `requirements.txt`
In addition, install baseline with the following command
`pip install git+https://github.com/openai/baselines #>`

Run train.py (from within the `code` folder) with 
`python train.py`

The arguments you can pass in are 

* `--num_envs` (int) number of parallel simulations in vectorized environment
* `--num_levels` (int) number of levels to train on
* `--timesteps` (int) number of timesteps to train
* `--save_interval` (int) how often to save (number of logs per checkpoint created) (if 0, doesn't save)
* `--log_dir` (str) where to save logs and checkpointstype=str, default='logs/')

These arguments can also be seen in detail by running `python train.py --help`



The various notebooks can be loaded in Colab, and then cells can be run sequentially. Note that many of the paths is particular to the layout of one's Drive, and there are many experiments.# fruitbot
