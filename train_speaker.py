# env = simple_speaker_listener_v4.parallel_env(render_mode="human")
# observations, infos = env.reset()

# print(observations)

# # while env.agents:
#     # this is where you would insert your policy
#     # actions = {agent: env.action_space(agent).sample() for agent in env.agents}

#     # observations, rewards, terminations, truncations, infos = env.step(actions)
# # env.close()


import numpy as np
from buffer import ReplayBuffer
from matd3 import MATD3
from pettingzoo.mpe import simple_adversary_v3

env = simple_adversary_v3.parallel_env(continuous_actions=True)
env.reset()
agent_names = env.agents
actor_dims = [env.observation_space(agent_name).shape[0] for agent_name in agent_names]
action_dims = [env.action_space(agent_name).shape[0] for agent_name in agent_names]

matd3_agent = MATD3(agent_names, actor_dims, action_dims)
buffer = ReplayBuffer(10000, actor_dims, action_dims, batch_size=1000, agent_names=agent_names)


N_GAMES = 100000
MAX_STEP_PER_EPISODE = 75
global_step = 0

for i in range(N_GAMES):
    state, info = env.reset()
    actions = {agent : env.action_space(agent).sample() for agent in agent_names}
    next_state, reward, termination, truncation, _ = env.step(actions)
    env.step(actions)

    critic_state = np.concatenate([s for s in state.values()])

    score = 0
    done = [False] * len(agent_names)
    episode_step = 0

    while not any(done):
        actions = matd3_agent.choose_action(state)
 
        next_state, reward, termination, truncation, _ = env.step(actions) 
        global_step += 1

        critic_next_state = np.concatenate([s for s in next_state.values()])

        score += sum(reward.values())
        episode_step += 1 

        if any(termination.values()) or any(truncation.values()):
            done = [True] * len(agent_names)
        buffer.store_transition(state, critic_state, actions, list(reward.values()), next_state, critic_next_state, done) 

        state = next_state
        critic_state = critic_next_state

        matd3_agent.update_critic(buffer)
        matd3_agent.update_target_critic()
        matd3_agent.update_target_actor()
        if (global_step+1)%2==0:
            matd3_agent.update_critic(buffer)

    print(f'Episode: {i+1} - score: {score} - num step: {episode_step}')