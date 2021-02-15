import logging
from gym.envs.registration import register

logger = logging.getLogger(__name__)

register(
    id='leo_power_att_env-v0',
    entry_point='basilisk_env.envs:leoPowerAttEnv'
)
register(
    id='earth_obs_env-v0',
    entry_point='basilisk_env.envs:EarthObsEnvironment'
)

# register(
#     id='opnav_env-v0',
#     entry_point='basilisk_env.envs:opNavEnv'
# )