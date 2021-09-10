
from setuptools import find_packages, setup
from pathlib import Path

this_directory = Path(__file__).parent
readme = (this_directory / "README.md").read_text()


setup(
    name='sentencesimilarity',
    packages=find_packages(),
    version='0.1.1',
    description='Calculates semantic similarity between given sentences.',
    long_description= readme,
    long_description_content_type='text/markdown',
    author='osahin',
    author_email = "oguuzhansahiin@gmail.com",
    license='MIT',
    install_requires=['transformers==4.9.2','scikit_learn==0.24.2','torch==1.9.0'],
    setup_requires=['pytest-runner'],
    tests_require=['pytest==4.4.1'],
    test_suite='tests',
)
