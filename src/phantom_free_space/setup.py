import os
from glob import glob

from setuptools import setup

package_name = 'phantom_free_space'

setup(
    name=package_name,
    version='0.0.1',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages', ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'config'), glob('config/*.yaml')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='phantom',
    maintainer_email='phantom@example.com',
    description='PHANTOM phase 1 local free-space feature extraction.',
    license='MIT',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'free_space_node = phantom_free_space.free_space_node:main',
        ],
    },
)
