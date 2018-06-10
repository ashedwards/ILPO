from gym.envs.registration import registry, register, make, spec

register(
    id='ThorFridge-v0',
    entry_point='gym_thor.thor_env:ThorEnv',
)

register(
    id='SimpleThorFridge-v0',
    entry_point='gym_thor.simple_thor_env:SimpleThorEnv',
)
