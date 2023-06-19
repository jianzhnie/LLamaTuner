'''
Author: jianzhnie
Date: 2021-09-30 15:23:37
LastEditTime: 2021-11-29 16:59:02
LastEditors: jianzhnie
Description:

'''
import os
import sys

from setuptools import find_packages, setup

if __name__ == '__main__':

    if sys.version_info < (3, 7):
        raise ValueError(
            'Unsupported Python version %d.%d.%d found. Auto-Timm requires Python '
            '3.7 or higher.' % (sys.version_info.major, sys.version_info.minor,
                                sys.version_info.micro))

    HERE = os.path.abspath(os.path.dirname(__file__))
    with open(os.path.join(HERE, 'requirements.txt')) as fp:
        install_reqs = [
            r.rstrip() for r in fp.readlines()
            if not r.startswith('#') and not r.startswith('git+')
        ]

    with open('chatllms/__version__.py') as fh:
        version = fh.readlines()[-1].split()[-1].strip("\"'")

    with open('README.md', encoding='utf-8') as fh:
        long_description = fh.read()

    setup(
        name='chatllms',
        author='Jianzh Nie',
        author_email='jianzhnie@gmail.com',
        description='chatllms is a open source chatbot framework.',
        long_description=long_description,
        long_description_content_type='text/markdown',
        version=version,
        packages=find_packages(exclude=[
            'assets', 'benchmark', 'docs', 'examples', 'test', 'scripts',
            'tools'
        ]),
        install_requires=install_reqs,
        include_package_data=True,
        license='Apache License',
        platforms=['Linux'],
        classifiers=[
            'Environment :: Console',
            'Intended Audience :: Developers',
            'Intended Audience :: Education',
            'Intended Audience :: Science/Research',
            'Intended Audience :: Information Technology',
            'Natural Language :: English',
            'Operating System :: OS Independent',
            'Topic :: Scientific/Engineering :: Artificial Intelligence',
            'Topic :: Scientific/Engineering :: Information Analysis',
            'Programming Language :: Python :: 3.7',
            'Programming Language :: Python :: 3.8',
            'Programming Language :: Python :: 3.9',
        ],
        keywords=['chatllms', 'rlhf', 'llm'],
        python_requires='>=3.7',
        url='https://github.com/jianzhnie/Efficient-Tuning-LLMs',
    )
