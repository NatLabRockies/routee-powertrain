# %%
from routee.powertrain.registry import LocalRegistry 
# %%
registry = LocalRegistry("../model-library.ignore")
# %%
registry.query(make="chevrolet")
# %%
