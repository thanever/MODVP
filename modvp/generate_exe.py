# -*- coding: utf-8 -*-
"""
Created on Tue Mar 31 21:55:41 2009

@author: T.Han
"""

from distutils.core import setup
import py2exe

py2exe_options = {
        "includes":["sip",],
        }

setup(windows=["DVP_ui.py"], options={'py2exe':py2exe_options})