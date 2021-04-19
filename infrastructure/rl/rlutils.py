from tensorflow.python.data import Dataset


# The goal for this is to simplify the conversion of transitions in an RL environment into a dataset.
# Ideally, one need not track the initial and final states separately, so the state sequence should
# contain one more entry than the action and reward sequences.  Rewards should be discounted easily
# and automatically
class TransitionDataset(Dataset):

    def __init__(self, state_seq, action_seq, reward_seq):
        pass



#
class AutoencoderDataset(Dataset):

    def __init__(self, dataset):
        self.dataset = dataset

