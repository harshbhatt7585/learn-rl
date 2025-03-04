"""
First, we will define a set of hyperparameters we will be using for training.

Next, we will focus on creating our environment, or simulator, using TorchRL’s wrappers and transforms.

Next, we will design the policy network and the value model, which is indispensable to the loss function. These modules will be used to configure our loss module.

Next, we will create the replay buffer and data loader.

Finally, we will run our training loop and analyze the results.
"""

import warnings
warnings.filterwarnings("ignore")
from torch import multiprocessing


from collections import defaultdict

import matplotlib.pyplot as plt
import torch
from tensordict.nn import TensorDictModule
from tensordict.nn.distributions import NormalParamExtractor
from torch import nn
from torchrl.collectors import SyncDataCollector
from torchrl.data.replay_buffers import ReplayBuffer
from torchrl.data.replay_buffers.samplers import SamplerWithoutReplacement
from torchrl.data.replay_buffers.storages import LazyTensorStorage
from torchrl.envs import (Compose, DoubleToFloat, ObservationNorm, StepCounter,
                          TransformedEnv)
from torchrl.envs.libs.gym import GymEnv
from torchrl.envs.utils import check_env_specs, ExplorationType, set_exploration_type
from torchrl.modules import ProbabilisticActor, TanhNormal, ValueOperator
from torchrl.objectives import ClipPPOLoss
from torchrl.objectives.value import GAE
from tqdm import tqdm

is_fork = multiprocessing.get_start_method() == "fork"
device = (
    torch.device(0)
    if torch.cuda.is_available() and not is_fork
    else torch.device("cpu")
)
num_cells = 256  # number of cells in each layer i.e. output dim.
lr = 3e-4
max_grad_norm = 1.0

frames_per_batch = 1000
# For a complete training, bring the number of frames up to 1M
total_frames = 50_000

sub_batch_size = 64
num_epochs = 10
num_epochs = 10
clip_epsilon = (
    0.2
)
gamma = 0.99
lmbda = 0.95
entropy_eps = 1e-4


base_env = GymEnv("InvertedDoublePendulum-v4", device=device)

env = TransformedEnv(
    base_env,
    Compose(
        ObservationNorm(in_keys=["observation"]),
        DoubleToFloat(),
        StepCounter()
    )
)

env.transform[0].init_stats(num_iter=1000, reduce_dim=0, cat_dim=0)
print("normalization constant shape:", env.transform[0].loc.shape)
print("observation_spec:", env.observation_spec)
print("reward_spec:", env.reward_spec)
print("input_spec:", env.input_spec)
print("action_spec (as defined by input_spec):", env.action_spec)
check_env_specs(env)


rollout = env.rollout(3)
print("rollout of three steps:", rollout)
print("Shape of the rollout TensorDict:", rollout.batch_size)



"""
Define a neural network D_obs -> 2 * D_action. Indeed, our loc (mu) and scale (sigma) both have dimension D_action.

Append a NormalParamExtractor to extract a location and a scale (for example, splits the input in two equal parts and applies a positive transformation to the scale parameter).

Create a probabilistic TensorDictModule that can generate this distribution and sample from it.
"""

actor_net = nn.Sequential(
    nn.LazyLinear(num_cells, device=device),
    nn.Tanh(),
    nn.LazyLinear(num_cells, device=device),
    nn.Tanh(),
    nn.LazyLinear(num_cells, device=device),
    nn.Tanh(),
    nn.LazyLinear(2 * env.action_spec.shape[-1], device=device),
    NormalParamExtractor()
)

policy_module = TensorDictModule(
    actor_net, in_keys=['observation'], out_keys=["loc", "scale"]
)


policy_module = ProbabilisticActor(
    module=policy_module,
    spec=env.action_spec,
    in_keys=["loc", "scale"],
    distribution_class=TanhNormal,
    distribution_kwargs={
        "low": env.action_spec.space.low,
        "high": env.action_spec.space.high,
    },
    return_log_prob=True
)

value_net = nn.Sequential(
    nn.LazyLinear(num_cells, device=device),
    nn.Tanh(),
    nn.LazyLinear(num_cells, device=device),
    nn.Tanh(),
    nn.LazyLinear(num_cells, device=device),
    nn.Tanh(),
    nn.LazyLinear(1, device=device)
)

value_module = ValueOperator(
    module=value_net,
    in_keys=["observation"]
)

print("Running policy:", policy_module(env.reset()))
print("Running value:", value_module(env.reset()))


collector = SyncDataCollector(
    env,
    policy_module,
    frames_per_batch=frames_per_batch,
    total_frames=total_frames,
    split_trajs=False,
    device=device
)


replay_buffer = ReplayBuffer(
    sotrage=LazyTensorStorage(max_size=frames_per_batch),
    sampler=SamplerWithoutReplacement(),
)

 
advantage_module = GAE(
    gamma=gamma, lmbda=lmbda, value_network=value_module, average_gae=True
)

loss_module = ClipPPOLoss(
    actor_network=policy_module,
    critic_network=value_module,
    clip_epsilon=clip_epsilon,
    entropy_bonus=bool(entropy_eps),
    entropy_coef=entropy_eps,
    critic_coef=1.0,
    loss_critic_type="smooth_l1"
)

optim = torch.optim.Adam(loss_module.parameters(), lr)
scheduler = torch.optim.lr.scheduler.CosineAnnealingLr(
    optim,
    total_frames//frames_per_batch,
    0.0
)



## Training Loop

"""
Collect data

    Compute advantage

        Loop over the collected to compute loss values

        Back propagate

        Optimize

        Repeat

    Repeat

Repeat
"""

logs = defaultdict(list)
pbar = tqdm(total=total_frames)
eval_str= ""

for i, tensordict_data in enumerate(collector):
    for _ in range(num_epochs):

        advantage_module(tensordict_data)
        data_view = tensordict_data.reshape(1)
        replay_buffer.extend(data_view.cpu())
        for _ in range(frames_per_batch // sub_batch_size):
            subdata = replay_buffer.sample(sub_batch_size)
            loss_vals = loss_module(subdata.to(device))
            loss_value = (
                loss_vals["loss_objective"]
                + loss_vals["loss_critic"]
                + loss_vals["loss_entropy"]
            )

            loss_value.backward()
            torch.nn.utils.clip_grad_norm_(loss_module.parameters(), max_grad_norm)
            optim.step()
            optim.zero_grad()

        logs["reward"].append(tensordict_data["next"], "reward").mean().item()
        pbar.update(tensordict_data.numel())
        cum_reward_str = (
        f"average reward={logs['reward'][-1]: 4.4f} (init={logs['reward'][0]: 4.4f})"
        )
        logs["step_count"].append(tensordict_data["step_count"]).max().item()
        stepcount_str = f"step count (max): {logs["step_count"][-1]}"
        logs["lr"].append(optim.param_groups[0]["lr"])
        lr_str = f"lr policy: {logs['lr'][-1]: 4.4f}"

        # TODO: Evaluation Code

        scheduler.step()


    



# TODO: Plot Graph

        

