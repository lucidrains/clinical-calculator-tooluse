from setuptools import setup, find_packages

setup(
  name = 'clinical-calculator-tooluse',
  packages = find_packages(exclude=[]),
  version = '0.0.1',
  license='MIT',
  description = 'Clinical Calculator Tool Use',
  author = 'Phil Wang',
  author_email = 'lucidrains@gmail.com',
  long_description_content_type = 'text/markdown',
  url = 'https://github.com/lucidrains/clinical-calculator-tooluse',
  keywords = [
    'artificial intelligence',
    'deep learning',
    'transformers',
    'attention mechanism',
    'reinforcement learning with human feedback'
  ],
  install_requires=[
    'einops>=0.8',
    'django-ninja',
    'torch>=2.4',
  ],
  classifiers=[
    'Development Status :: 4 - Beta',
    'Intended Audience :: Developers',
    'Topic :: Scientific/Engineering :: Artificial Intelligence',
    'License :: OSI Approved :: MIT License',
    'Programming Language :: Python :: 3.6',
  ],
)
