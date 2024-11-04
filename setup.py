from setuptools import find_packages
from setuptools import setup

with open("requirements.txt") as f:
    content = f.readlines()
requirements = [x.strip() for x in content if "git+" not in x]

setup(name='detect_ai_content',
      version="0.0.1",
<<<<<<< HEAD
      description="Lewagon Final Project: Detect AI Content",
      install_requires=requirements,
      packages=find_packages(),
    )
=======
      description="Detect AI Content Model",
      license="MIT",
      author="",
      author_email="",
      url="https://github.com/yukaberry/detect_ai_content",
      install_requires=requirements,
      packages=find_packages(),
      test_suite="tests",
      # include_package_data: to install data from MANIFEST.in
      include_package_data=True,
      zip_safe=False)
>>>>>>> master
