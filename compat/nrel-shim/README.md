# nrel.routee.powertrain has moved

**This project is now published as [`routee.powertrain`](https://pypi.org/project/routee.powertrain/).**

The National Laboratory of the Rockies rename means the `nrel` namespace has been dropped. This
package is a tombstone: installing it pulls in `routee.powertrain`, and importing it raises an
`ImportError` telling you what to change.

```bash
pip uninstall nrel.routee.powertrain
pip install routee.powertrain
```

```diff
-import nrel.routee.powertrain as pt
+import routee.powertrain as pt
```

Version 2.0 is a breaking release beyond the rename — `Model`, `load_model`, and the on-disk model
format all changed. See the
[migration guide](https://natlabrockies.github.io/routee-powertrain/migrating_from_v1.html).

If you need the old code, `nrel.routee.powertrain==1.4.1` is the final 1.x release.
