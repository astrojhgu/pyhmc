#!/usr/bin/env python3
from setuptools import setup
import hmc

print("Setupping pyhmc version={0}".format(hmc.__version__))
setup(
    name="hmc",
    version=hmc.__version__,
    packages=["hmc"],
    zip_safe=True,
)
