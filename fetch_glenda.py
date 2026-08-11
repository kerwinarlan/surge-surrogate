import pandas as pd

# IBTrACS Western Pacific Dataset URL
IBTRACS_URL = "https://www.ncei.noaa.gov/data/international-best-track-archive-for-climate-stewardship-ibtracs/v04r01/access/csv/ibtracs.WP.list.v04r01.csv"

def fetch_glenda():
    print("Downloading IBTrACS Western Pacific track dataset...")
    # Line 1 in IBTrACS contains unit headers, so we skip row index 1
    df = pd.read_csv(IBTRACS_URL, skiprows=[1], low_memory=False)
    
    # Filter for Rammasun (Glenda) in 2014
    glenda = df[(df['NAME'].str.upper() == 'RAMMASUN') & (df['SEASON'].astype(str) == '2014')].copy()
    
    # Select key attributes: ISO Time, Lat, Lon, Wind Speed (kts), Central Pressure (hPa)
    cols = ['ISO_TIME', 'LAT', 'LON', 'USA_WIND', 'USA_PRES', 'WMO_WIND', 'WMO_PRES']
    glenda_track = glenda[cols]
    
    print(f"Found {len(glenda_track)} points for Typhoon Glenda.")
    glenda_track.to_csv("data/glenda_track_2014.csv", index=False)
    print("Successfully saved to data/glenda_track_2014.csv!")

if __name__ == "__main__":
    fetch_glenda()
