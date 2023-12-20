from setuptools import setup, find_packages

setup(
    name="slot_diffusion_policy",
    packages=['slot_diffusion_policy',
              'slot_diffusion_policy.model'
              'slot_diffusion_policy.model.slot',
              'slot_diffusion_policy.dataset',],
)