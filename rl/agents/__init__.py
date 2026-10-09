from rl.agents.baselines import FixedPolicy, RandomPolicy, RulePolicy
from rl.agents.dqn import DQN
from rl.agents.noisy_dueling_dqn import NoisyDuelingDQN

LEARNED = {"dqn": DQN, "noisy_dueling_dqn": NoisyDuelingDQN}
