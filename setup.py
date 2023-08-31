from setuptools import setup, find_packages
import os

# thelibFolder = os.path.dirname(os.path.realpath(__file__))
# requirementPath = thelibFolder + '/requirements.txt'

# install_requires = [] 
# if os.path.isfile(requirementPath):
#     with open(requirementPath) as f:
#         install_requires = f.read().splitlines()

with open("README.md", "r") as fh:
    long_description = fh.read()

setup(
	name='face_sampler',
	version='0.1',
	description='Extracts the face from images at a desired resolution',
	license='Apache',
	author='John Eicher',
	author_email='john.eicher89@gmail.com',
	packages=find_packages(),
	download_url='https://github.com/saltchicken/face_sampler',
	maintainer_email='john.eicher89@gmail.com',
	# install_requires=install_requires,
	long_description=long_description,
    long_description_content_type="text/markdown",
	classifiers=[
		'Development Status :: 1 - Beta',
		'Topic :: Software Development :: Build Tools'
	],
	entry_points ={ 
        'console_scripts': [ 
            'face_sampler = face_sampler:main'
        ] 
    },
    python_requires='>=3.5',
)