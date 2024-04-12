from setuptools import setup, find_packages

setup(

    # Other setup() args here

    platforms=['Linux'],
    python_requires='>=3.5',
    install_requires=[
        'python-apt>=2.0',
        # Add any other dependencies here
        'graph-tools>=1.5'
    ],
    package_dir={'': 'lib'},
    scripts=glob.glob('bin/*'),

    # Add other arguments as needed

)
