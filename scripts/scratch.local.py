# %%
from routee.powertrain.registry import LocalRegistry 
# %%
registry = LocalRegistry("../model-library.ignore")
# %%
registry.query(make="chevy", model_name="bolt")
# %%
model = registry.load(
    "chevrolet/bolt_ev/2020/steady/ambient_temp_f_grade_percent_speed_mph/v1"
)
# %%
model
# %%
