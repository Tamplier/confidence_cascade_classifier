from setuptools import setup, find_packages

setup(
    name="confidence_cascade",
    version="0.1.0",
    description="Cascade classifier that routes uncertain predictions to deeper models.",
    author="Alex Tinekov",
    author_email='araman777@gmail.com',
    packages=find_packages(),
    install_requires=[
        "numpy",
        "scikit-learn"
    ],
    python_requires=">=3.10",
)
