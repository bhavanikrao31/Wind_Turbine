import numpy as np
import pandas as pd

rng = np.random.default_rng(42)
n = 4320  # 30 days of 10-minute data
time = pd.date_range("2026-09-01", periods=n, freq="10min")

def power_curve(v):
    p = 2000 * ((v - 3) / 9) ** 3
    p = np.where(v < 3, 0, p)
    p = np.where(v >= 12, 2000, p)
    p = np.where(v >= 25, 0, p)
    return p

# turbine: (power loss, rpm factor, extra temp)
faults = {
    1: (0.00, 1.0, 0), 2: (0.01, 1.0, 0), 3: (0.02, 1.0, 0),
    4: (0.12, 1.0, 20),   # gearbox friction
    5: (0.25, 0.70, 0),   # pitch fault / icing
    6: (0.08, 1.0, 0),    # sensor drift
    7: (0.35, 0.65, 0),   # severe icing
    8: (0.01, 1.0, 0),
}

frames = []
for i, (loss, rpm_f, temp_add) in faults.items():
    v = np.clip(rng.weibull(2.0, n) * 8, 0, 30)
    theo = power_curve(v)
    act = theo * (1 - loss) + rng.normal(0, 25, n) * (theo > 0)
    act = np.clip(act, 0, None)
    rpm = np.where(v >= 3, np.clip(1.3 * v, 0, 15.5), 0) * rpm_f
    rpm = np.clip(rpm + rng.normal(0, 0.3, n), 0, None)
    temp = 35 + 30 * (act / 2000) + temp_add + rng.normal(0, 1.5, n)
    frames.append(pd.DataFrame({
        "Timestamp": time, "Turbine_ID": f"T{i}",
        "WindSpeed": v.round(2), "Theoretical_Power": theo.round(2),
        "ActivePower": act.round(2), "Rotor_RPM": rpm.round(2),
        "Generator_Temp": temp.round(2)}))

df = pd.concat(frames, ignore_index=True)

# add messy data on purpose
df.loc[df.sample(frac=0.02, random_state=1).index, "Rotor_RPM"] = np.nan
df.loc[df.sample(frac=0.015, random_state=2).index, "Generator_Temp"] = np.nan
spikes = df.sample(40, random_state=4).index
df.loc[spikes, "ActivePower"] = df.loc[spikes, "ActivePower"] * 1.6 + 300
df = pd.concat([df, df.sample(20, random_state=3)], ignore_index=True)

df.to_csv("wind_scada.csv", index=False)
print("Done! Rows:", len(df))
