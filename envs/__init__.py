from gym.envs.registration import register

register(
    id='FFMP-v0',
    entry_point='gym_ffmp.envs:FFMP',
)
