from setuptools import setup, find_packages

setup(
    name="natify",
    version="1.0.0",
    author="Your Name",
    author_email="your.email@example.com",
    description="Indonesian Pronunciation App with ML-powered feedback",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/natify",
    packages=find_packages(),
    include_package_data=True,
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.9",
    install_requires=[
        line.strip() for line in open("requirements.txt") if not line.startswith("#")
    ],
    entry_points={
        "console_scripts": [
            "natify=app.main:main",
        ],
    },
)
