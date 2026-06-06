# Probabilistic Ensembles

Meteorological forecasting is inherently sensitive to initial condition uncertainties. To represent this, WeatherGraph provides a high-performance **Probabilistic Ensemble** inference engine. This engine runs multiple perturbed forecasts and aggregates statistics in real time using a memory-efficient $O(1)$ algorithm.

---

## 1. Noise Injection & Perturbation

To simulate chaotic fluctuations in the atmosphere, WeatherGraph injects additive Gaussian noise to dynamic coordinates at the start of each autoregressive step.

For each ensemble member $m$ at step $t$:

$$X_m^{(t+1)} = \text{Model}\left(X_m^{(t)}\right) + \epsilon_m, \quad \epsilon_m \sim \mathcal{N}(0, \sigma^2)$$

### Perturbation Scale Configuration
You can configure the standard deviation ($\sigma$) per variable using a dictionary:

```python
perturbation_scale = {
    "t": 0.05,  # 0.05 Kelvin perturbation for temperature
    "u": 0.1,   # 0.1 m/s perturbation for eastward wind
    "v": 0.1,   # 0.1 m/s perturbation for northward wind
    "q": 0.0001 # 1e-4 kg/kg perturbation for humidity
}
```

---

## 2. Tested Python Example

The code snippet below demonstrates how to run an ensemble forecast in Python. This example is verified in the test suite:

```python
--8<-- "tests/doc_examples/test_ensembles.py:ensemble_prediction"
```

---

## 3. Memory Optimization: Welford's Algorithm

Running standard ensemble forecasts requires storing every time slice for every member, which consumes vast amounts of RAM. WeatherGraph avoids this by calculating the mean, variance, and standard deviation in-place using **Welford's Algorithm**.

As each member $m$ completes its rollout, the C++ engine updates the statistics accumulators step-by-step:

$$M_m = M_{m-1} + \frac{x_m - M_{m-1}}{m}$$

$$S_m = S_{m-1} + (x_m - M_{m-1})(x_m - M_m)$$

Once all members have completed, the standard deviation is computed:

$$\sigma = \sqrt{\frac{S_M}{M}}$$

This approach guarantees that the memory footprint is constant and depends only on the size of a single grid slice, regardless of whether you run 10, 50, or 1000 ensemble members.

---

## 4. Rule-Based Threshold Probabilities

You can define custom threshold rules to evaluate the probability of specific meteorological events (e.g. frost risk).

### Expression Syntax
Threshold expressions follow a simple format:
*   `t@850 < 273.15`: Evaluates temperature at the 850 hPa vertical level.
*   `q < 0.005`: Evaluates specific humidity across all 13 vertical levels.

The engine parses these strings and accumulates the percentage of members that satisfy each rule:

```python
thresholds = {
    "frost_risk": "t@1000 < 273.15",
    "extreme_wind": "u@850 > 25.0"
}
```

The resulting `stats.probabilities` dictionary contains spatial arrays where values range from `0.0` (no member met the criteria) to `1.0` (all members met the criteria), representing spatial forecast probability maps.

---

## 5. Running Ensembles via the CLI

You can execute the same ensemble forecasts from the command line:

```bash
weathergraph ensemble \
  --model-path models/weather_gnn.onnx \
  --weights-dir data \
  --data-source era5_netcdf \
  --input-path data/era5_archives/init.nc \
  --steps 40 \
  --members 50 \
  --perturbation-scale '{"t": 0.05, "u": 0.1}' \
  --threshold frost=t@850<273.15 \
  --output-format netcdf4 \
  --output-path output/ensemble_output
```
This command generates `ensemble_mean.nc`, `ensemble_std_dev.nc`, and `prob_frost.nc` files inside the output directory.