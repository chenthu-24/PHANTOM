from setuptools import setup

package_name = 'phantom_sensor_bridge'

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
    description='PHANTOM phase 1/2 fake sensor bridge for local validation.',
    license='MIT',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'sensor_bridge_node = phantom_sensor_bridge.sensor_bridge_node:main',
        ],
    },
)
