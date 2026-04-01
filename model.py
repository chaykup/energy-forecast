from nixtla import NixtlaClient
import os
from dotenv import load_dotenv
import pandas as pd

load_dotenv()
API_KEY = os.getenv('NIXTLA_API_KEY')

nixtla_client = NixtlaClient(
    api_key=API_KEY
)

# nixtla_client.validate_api_key()

df = pd.read_csv('air_passengers.csv')

timegpt_fcst_df = nixtla_client.forecast(
    df=df,
    h=12,
    freq='MS',
    time_col='timestamp',
    target_col='value'
)

fig = nixtla_client.plot(df, timegpt_fcst_df, time_col='timestamp', target_col='value')
fig.savefig('plot2.png', bbox_inches='tight')