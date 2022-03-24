from setuptools import setup, dist

# Python build requirements
requirements = []

class BinaryDistribution(dist.Distribution):
    def is_pure(self):
        return False

setup(
    name='rusty_tree',
    version='0.0.1',
    descripion='Python Bindings for Rusty Trees',
    packages=['rusty_tree'],
    include_package_data=True,
    package_data={
        'rusty_tree': ['lib/librusty_tree.so']
    },
    distclass=BinaryDistribution,
    zip_safe=False,
)