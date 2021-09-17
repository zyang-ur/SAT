from setuptools import setup

setup(name='SAT',
      version='0.1',
      description='SAT 2D Semantics Assisted Training for 3D Visual Grounding.',
      url='https://github.com/zyang-ur/SAT',
      author='zyang',
      author_email='zhengyuan.yang13@gmail.com',
      license='MIT',
      install_requires=['scikit-learn',
                        'matplotlib',
                        'six',
                        'tqdm',
                        'pandas',
                        'plyfile',
                        'requests',
                        'symspellpy',
                        'termcolor',
                        'tensorboardX',
                        'shapely',
                        'pyyaml'
                        ],
      packages=['SAT'],
      zip_safe=False)
