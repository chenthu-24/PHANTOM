from setuptools import setup

package_name = 'phantom_subgoal_generator'

setup(
    name=package_name,
    version='0.0.1',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages', ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='phantom',
    maintainer_email='phantom@example.com',
    description='PHANTOM phase 2 tactical local subgoal generator.',
    license='MIT',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'subgoal_generator_node = phantom_subgoal_generator.subgoal_generator_node:main',
        ],
    },
)
