from setuptools import setup, find_packages

README = open('README.md').read()
REQUIREMENTS = open('requirements.txt').read().splitlines()
print(REQUIREMENTS)

# Optional dependencies
REQUIREMENTS_ZERO_SHOT = open(
    'requirements_zero_shot.txt').read().splitlines()
REQUIREMENTS_ALL = REQUIREMENTS + REQUIREMENTS_ZERO_SHOT

setup(name='vjp',
      classifiers=['Intended Audience :: Developers',
                   'Programming Language :: Python :: 3 :: Only'],
      python_requires='>=3.9',
      version='0.9.0',
      description='VAT Judgement Prediction on italianVAT',
      long_description_content_type='text/markdown',
      long_description=README,
      url='https://github.com/Ball-Man/vjp-ita',
      # author='Francesco Mistri',
      # author_email='franc.mistri@gmail.com',
      license='MIT',
      packages=find_packages(),
      install_requires=REQUIREMENTS,
      extras_require={
        'zeroshot': REQUIREMENTS_ZERO_SHOT}
      )
