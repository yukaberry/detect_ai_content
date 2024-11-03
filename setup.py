from setuptools import find_packages
from setuptools import setup

with open("requirements.txt") as f:
    content = f.readlines()
requirements = [x.strip() for x in content if "git+" not in x]

setup(name='detect_ai_content',
      version="0.0.1",
      description="Lewagon Final Project: Detect AI Content",
      install_requires=requirements,
      packages=find_packages(),
    )
