from setuptools import setup

package_name = 'phantom_mode_manager'

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
    description='PHANTOM phase 2 tactical mode state machine.',
    license='MIT',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'mode_manager_node = phantom_mode_manager.mode_manager_node:main',
        ],
    },
)
