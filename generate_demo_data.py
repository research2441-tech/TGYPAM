"""Deterministic synthetic data for software smoke tests; NOT study observations."""
import argparse
import numpy as np
import pandas as pd


def generate(n_fields=40, months=60, seed=42):
    rng = np.random.default_rng(seed)
    crops = ['rice', 'wheat', 'maize', 'groundnut']
    rows = []
    for field in range(n_fields):
        crop = crops[field % len(crops)]
        soil = rng.uniform(5.5, 7.8)
        base_n = rng.uniform(30, 75)
        for t in range(months):
            temp = 22 + 6*np.sin(2*np.pi*t/12) + rng.normal(0, 2)
            humid = float(np.clip(65 + 15*np.cos(2*np.pi*t/12)+rng.normal(0, 5),20,100))
            rainfall = max(0, rng.gamma(2,12) + 12*np.cos(2*np.pi*t/12))
            irrigation = max(0, 45-rainfall + rng.normal(0,4))
            moisture = float(np.clip(25+.25*rainfall+.2*irrigation+rng.normal(0,4),5,100))
            nutrients = base_n + rng.normal(0,3)
            price = max(40, (260 + 40*(field%4)) * (1 + .01*t) + rng.normal(0,20))
            costs = 480 + 2*irrigation + rng.uniform(0,110)
            growth = float(np.clip(.22+.005*nutrients +.003*moisture-.003*abs(temp-27)+rng.normal(0,.06),0,1))
            yield_t = max(0, .2 + 3.3*growth+.008*rainfall+rng.normal(0,.3))
            profit = yield_t*price-costs
            water = irrigation+rainfall
            fertilizer = max(0, 95-.5*nutrients+rng.normal(0,5))
            rows.append(dict(timestamp=str((pd.Timestamp('2021-01-01')+pd.DateOffset(months=t)).date()),field_id=f'field_{field:04d}',crop=crop,temperature_c=temp,humidity_pct=humid,rainfall_mm=rainfall,soil_ph=soil,soil_n=nutrients,soil_p=14+field%12,soil_k=55+field%18,irrigation_mm=irrigation,soil_moisture_pct=moisture,sunlight_hours=7+2*np.sin(t/3),wind_kmh=8+field%6,field_area_ha=1+field%4,market_price_usd_t=price,input_cost_usd_ha=costs,growth=growth,yield_t_ha=yield_t,profit_usd_ha=profit,water_mm=water,fertilizer_kg_ha=fertilizer))
    return pd.DataFrame(rows)

if __name__ == '__main__':
    p=argparse.ArgumentParser();p.add_argument('--fields',type=int,default=40);p.add_argument('--months',type=int,default=60);p.add_argument('--out',default='agriculture.csv');a=p.parse_args()
    if a.months<12: raise ValueError('months must be >= 12')
    df=generate(a.fields,a.months);df.to_csv(a.out,index=False)
    print(f'SYNTHETIC ONLY: wrote {len(df)} observations to {a.out}')
