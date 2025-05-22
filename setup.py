from setuptools import setup, find_packages

setup(
    name='cuda_sgp4',
    version='0.1.0',
    packages=find_packages(),
    install_requires=[
        'numpy',
        'numba',
        # Add other dependencies here
    ],
    author='Your Name',
    author_email='your.email@example.com',
    description='CUDA-accelerated SGP4 orbit propagation package',
    url='https://github.com/yourusername/cuda_sgp4',  # Replace with your repository URL
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.6',
)
