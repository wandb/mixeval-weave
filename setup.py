from setuptools import setup, find_packages

setup(
    name='mixeval-weave',
    version='0.1',
    author='Ayush Thakur',
    author_email='ayusht@wandb.com',
    description='Weave implementation of MixEval',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    url='https://github.com/wandb/mixeval-weave',
    packages=find_packages(),
    install_requires=[
        # Add your package dependencies here
    ],
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.10.12',
)