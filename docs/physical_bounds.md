# Physical Bounds and Model Caveats

The models in the RouteE Powertrain library are statistical fits, not physical
simulations. They are good at what they were fit to and can be wrong
outside it. This page attempts to make clear where the boundaries are
and how to exercise caution when using the models.

## Know what these models can and cannot do

The random forest, CNN and NGBoost models were trained on simulated (FASTSim)
energy over short road links: roughly 0.01 to 0.5 miles with most average speeds falling between 5 to 70 mph.
Outside that range, above all on links longer than half a mile, a prediction is
extrapolation and can be incorrect.

**Trip totals over real road links are reliable.** Individual link predictions
are noisier, and they are not guaranteed to obey physics. On a single link a
model may charge less for a climb than the height gained, return more energy on
a descent than the hill contains, or price a hill and its return leg below flat
ground.

**CNN models read the previous four links of a trip.** Give them whole trips in
driving order. A lone link, or a trip with links shuffled or missing, is not the
input they were trained on.

Run `routee-powertrain validate-physics <model>` to see which physical checks a
specific model passes on a synthetic sweep of links.

## The output guardrail

Since 2.1.0, `Model.predict` clips every link's energy to a **physical
ceiling** before returning it: the energy needed to lift the vehicle up the
link's rise, bring it to the link's speed once, and push it against rolling and
aerodynamic resistance over the link's length, plus an
accessory draw for the time the link takes. For a fuel target it also clips at
zero: burned fuel does not return to the tank.

The vehicle constants behind the ceiling are deliberately generous (a heavy
rolling resistance, a large drag area, a poor driveline), so it is a bound on
any plausible vehicle rather than an estimate for this one. It is the same
ceiling `validate-physics` reports as `absolute_ceiling`.

### What it does not do

- It does not fix the learned function. A climb can still cost less than its
  potential energy, a hill can still be cheaper than flat ground, and a
  BEV model can still return more on a descent than the hill held.
- It needs a vehicle mass (`vehicle.mass_lbs` in the metadata, or a mass
  feature), a speed feature, a distance in miles and a target in a recognised
  energy unit. When any is missing the prediction passes through unchanged.
- It applies to `Model.predict` and to lookup tables. A consumer that runs the
  estimator binary itself gets raw output.

### Turning it off

The `contract.output_guardrail` field in `metadata.json` takes two values:
`"envelope"` (the default, and what every model published before the field
existed loads as) or `"none"`. Set it at training time on the `ModelConfig`
for an estimator whose outputs are bounded by construction, or at runtime:

```python
model = pt.load_model("chevrolet/bolt_bev/2017/rf_base_fe510e40")
model.metadata.contract.output_guardrail = "none"
```

Raw output, with neither the clamp nor the adjustment factor, is one call away:

```python
raw = model.estimator.predict(links_df, model.metadata.config)
```

And the band itself is public, so a caller can see where the clamp would bind:

```python
floor, ceiling = pt.physical_bounds(links_df, model.metadata.config)["electric_kwh"]
```

The floor is the validation floor described above; the guardrail applies only
the ceiling, and zero for fuel.
