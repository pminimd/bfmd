from setuptools import setup, find_packages

setup(
  name = 'bfmd',
  packages = find_packages(),
  version = '0.0.3',
  license='Apache Software License',
  description='Backbones for mobile device: feature extractor for real-time tasks',
  author = 'pminimd',
  author_email = 'pminimd@gmail.com',
  url = 'https://github.com/pminimd/bfmd',
  keywords = [
    'artificial intelligence',
    'computer vision',
    'Real-Time Neural Networks'
  ],
  install_requires=[
    'torch>=1.4',
    'torchvision',
  ],
  classifiers=[
      'Intended Audience :: Developers',
      'Intended Audience :: Science/Research',
      'License :: OSI Approved :: MIT License',
      'Programming Language :: Python :: 3.6',
      'Programming Language :: Python :: 3.7',
      'Programming Language :: Python :: 3.8',
      'Topic :: Scientific/Engineering :: Artificial Intelligence',
      'Topic :: Software Development',
      'Topic :: Software Development :: Libraries',
      'Topic :: Software Development :: Libraries :: Python Modules',
  ],
)