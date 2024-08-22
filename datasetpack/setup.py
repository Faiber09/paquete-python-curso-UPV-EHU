from setuptools import setup, find_packages

setup(
    name='datasetpack',
    version='0.0.1',
    author='Faiber Alonso',
    author_email='falonso010@ikasle.ehu.eus',
    packages=['datasetpack', 'datasetpack.test'],
    url='https://github.com/Faiber09', 
    license='LICENSE.txt',
    description='This package includes some basic functions to work with datasets, including normalization, standardization, discretization, and plotting.',
    long_description=open('README.txt').read(),
    tests_require=['pytest'],
    install_requires=[
        "seaborn >= 0.9.0",
        "pandas >= 0.25.1",
        "matplotlib >= 3.1.1",
        "numpy >=1.17.2"
    ],
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.6',
)