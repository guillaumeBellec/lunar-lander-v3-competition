import gymnasium as gym

ENV_KWARGS = dict(
    continuous=False,
    gravity=-7.5,
    enable_wind=True,
    wind_power=12.0,
    turbulence_power=1.2,
    max_episode_steps=1000,
)


def make_env(render_mode=None):
    return gym.make("LunarLander-v3", render_mode=render_mode, **ENV_KWARGS)
