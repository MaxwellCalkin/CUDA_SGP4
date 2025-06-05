from setuptools import setup, find_packages

setup(
    name='cuda_sgp4',
    version='0.1.0',
    packages=find_packages(),
    install_requires=[
        'numpy>=1.24.0',
        'numba>=0.59.1',
        'astropy>=5.2.1',
    ],
    tests_require=[
        'pytest>=8.3.3',
    ],
    author='Max Calkin',
    author_email='mcalkin@intaptai.com',
    description='CUDA-accelerated SGP4 orbit propagation package',
    url='https://github.com/maxwellcalkin/cuda_sgp4',  # Replace with your repository URL
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.10',
)
