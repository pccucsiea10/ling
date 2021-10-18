#!/usr/bin/env python
# coding: utf-8

# In[ ]:


class Config(object):
    pass

class ProdConfig(Config):
    pass

class DevConfig(Config):
    DEBUG = True

