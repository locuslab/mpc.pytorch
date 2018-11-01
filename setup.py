from setuptools import find_packages, setup

setup(
    name='mpc',
    version='0.0.3',
    description="A fast and differentiable MPC solver for PyTorch.",
    author='Brandon Amos',
    author_email='brandon.amos.cs@gmail.com',
    platforms=['any'],
    license="Apache 2.0",
    url='https://github.com/locuslab/mpc.pytorch',
    packages=find_packages(),
    install_requires=[
        'numpy>=1<2',
    ]
)
