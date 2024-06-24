# Code and Workspace convention!

## Using of Code Version Control
We are using `git` as our version control, the link to our orgainisation and repositories is https://github.com/Action-in-Quantum-Time.

For how to use `git`, please follow the guides (https://discord.com/channels/1239207624134754414/1246106715896938617/1246689597052686357) above.


## Using of Python and Packages

Just because the code is working on your machine does not mean it is guaranteed to with others, most of the time we can blame it on the differences between Python or package versions.
Thus the version of Python and its packages is very important, I have learned it the hard way with QISKIT, they change their APIs every 3 months, and redoing the code for compatibility can be frustrating. 

We will be using a specific version of Python, and specific version of packages to avoid confusions. We will be using two programs to manage all that: 

- Python Version Control: https://github.com/pyenv/pyenv
- Package Control https://pipenv.pypa.io/en/latest/

With these programs, we only do one command to install the compatible Python version plus the designated packages: `pipenv install`.

I will also set up the repositories in the orgainisation so that we can just clone and start working.~~~~

### Installation process

We will be using Python version 3.12 (We can decide on this later)
```shell
# Install python 3.12
pyenv install 3.12

# Set 3.12 as global
pyenv global 3.12
```

Assuming that you have already clonned the working repository, you may open a terminal within the repository and run `pipenv install` to (1) create and activate a python environment for the repository; (2) install all packages listed in `.pipenv` file.

Likewise, if you want to use `Conda`, you can find the Python Version in the file `.pyenv`.