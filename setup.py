from setuptools import setup, find_packages

if __name__ == '__main__':
    setup(
        name = 'trainer',
        version = '0.1',
        Author = 'Susum R',
        Author_email = 'susum@pluto7.com',
        Description = 'Training application package',
        packages = find_packages(include= 'trainer')
    )