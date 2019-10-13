import setuptools

# Get requirements from requirements.txt, stripping the version tags
with open('requirements.txt') as f:
    requires = f.readlines()

with open('README.md') as file:
    readme = file.read()

with open('HISTORY.md') as file:
    history = file.read()

setuptools.setup(name='plarx',
                 version='0.0.0',
                 description='Push numpy arrays through plugins',
                 author='Jelle Aalbers',
                 url='https://github.com/JelleAalbers/plarx',
                 long_description=readme + '\n\n' + history,
                 long_description_content_type="text/markdown",
                 setup_requires=['pytest-runner'],
                 install_requires=requires,
                 tests_require=requires + ['pytest'],
                 python_requires=">=3.6",
                 packages=setuptools.find_packages(),
                 classifiers=[
                     'Development Status :: 3 - Alpha',
                     'License :: OSI Approved :: BSD License',
                     'Natural Language :: English',
                     'Programming Language :: Python :: 3.6',
                     'Intended Audience :: Science/Research',
                     'Programming Language :: Python :: Implementation :: CPython',
                     'Topic :: Scientific/Engineering :: Physics',
                 ],
                 zip_safe=False)
