# %%
from routee.powertrain.registry import LocalRegistry 
# %%
registry = LocalRegistry("../model-library.ignore")
# %%
registry.query(make="chevy", model_name="bolt")
# %%
model = registry.load(
    "v2/chevrolet/bolt_ev/2017/default/grade_percent_speed_mph_turn_angle/v1"
)
# %%
model
# %%
