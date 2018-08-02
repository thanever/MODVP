# -*- coding: utf-8 -*-
"""
Created on Fri Jan 01 14:20:21 2016

@author: T.Han
"""
#
from distutils.core import setup
import py2exe
import sys
import zmq.libzmq
sys.setrecursionlimit(1000000)
#this allows to run it with a simple double click.
sys.argv.append('py2exe')
 
py2exe_options = {
        "includes": ["sip"],
        "dll_excludes": ["MSVCP90.dll",],
        "compressed": 1,
        "optimize": 1,
        "ascii": 0,
        }
 
setup(
      name = 'DVP_alpha',
      version = '1.0',
      windows = ['DVP.py',], 
      zipfile=None,
      options={
          'py2exe': {
              'includes': ['zmq.backend.cython'],
              'excludes': ['zmq.libzmq'],
              'dll_excludes': ['libzmq.pyd'],
          }
      },
      data_files=[
          ('lib', (zmq.libzmq.__file__,))
      ]
        )

