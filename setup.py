from setuptools import setup

setup(name='gunter',
      version='0.1',
      description='Algorithms related to Data Science',
      url='https://github.com/tbonza/gunter.git',
      author='tbonza',
      author_email='tylers.pile@gmail.com',
      license='MIT',
      packages=['gunter'],
      install_requires=[
          'numpy',
      ],
      setup_requires=['pytest-runner'],
      test_requires=[
          'pytest',
      ],
      test_suite="pytest",
      scripts=[],
      zip_safe=False)
