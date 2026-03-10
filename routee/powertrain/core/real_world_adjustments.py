from routee.powertrain.core.powertrain_type import PowertrainType

# real world adjustment factors used to generally correct
# for environmental variables (like temperature)
ADJUSTMENT_FACTORS = {
    PowertrainType.UNDEFINED: 1,
    PowertrainType.ICE: 1.166,
    PowertrainType.HEV: 1.1252,
    PowertrainType.BEV: 1.3958,
    PowertrainType.PHEV_EV_MODE: 1.3958,
    PowertrainType.PHEV_HEV_MODE: 1.1252,
    PowertrainType.HEAVY_DUTY: 1.0,
}
