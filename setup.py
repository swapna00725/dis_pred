from setuptools import find_packages,setup
from typing import List

a='-e .'

def get_requirements(file_path:str)->list[str]:
    requirements=[]
    with open(file_path) as file_obj :
        req=file_obj.readlines()
        req=[req1.replace('\n','') for req1 in req]
        if a in req:
            req.remove(a)
        return req
setup(
    name='disease_pred',
    version='0.0.1',
    author='swapna',
    author_email='mswapnaj@gmail.com',
    packages=find_packages(),
    install_requires=get_requirements('requirements.txt')
)