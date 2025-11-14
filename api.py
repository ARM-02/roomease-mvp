import requests
import base64
import pandas as pd
import json


class IdealistaAPIClient:
    # Class-level constants
    API_KEY = 'pway3lkvmj5wwgmzhfvvdk67btfnlggz'
    CLIENT_SECRET = '2slE9bRRQ295'
    BASE_URL = 'https://api.idealista.com/3.5/'
    COUNTRY = 'es'
    LANGUAGE = 'en'
    TOKEN_URL = "https://api.idealista.com/oauth/token"

    def __init__(self, max_items=50, operation="sale", property_type="homes", order="priceDown",
                 locationId='0-EU-ES-28-07-001-079-04-002', sort="desc", maxprice="1000000", minprice="1000"):
        self.max_items = max_items
        self.operation = operation
        self.property_type = property_type
        self.order = order
        self.locationId = locationId
        self.sort = sort
        self.maxprice = maxprice
        self.minprice = minprice
        self.access_token = None

        # Obtain token upon initialization
        self.get_access_token()

    def get_access_token(self):
        """Obtain and set the access token using the client credentials."""
        credentials = f"{self.API_KEY}:{self.CLIENT_SECRET}"
        encoded_credentials = base64.b64encode(credentials.encode()).decode()
        headers = {
            "Authorization": f"Basic {encoded_credentials}",
            "Content-Type": "application/x-www-form-urlencoded;charset=UTF-8"
        }
        data = {"grant_type": "client_credentials", "scope": "read"}
        response = requests.post(self.TOKEN_URL, data=data, headers=headers)

        if response.status_code == 200:
            self.access_token = response.json()["access_token"]
            print("Access token acquired.")
        else:
            raise Exception(f"Failed to get access token: {response.status_code}, {response.text}")

    def define_search_url(self, page):
        """Generate the search URL with the specified parameters."""
        url = (self.BASE_URL +
               f"{self.COUNTRY}/search?operation={self.operation}" +
               f"&maxItems={self.max_items}" +
               f"&order={self.order}" +
               f"&locationId={self.locationId}" +
               f"&propertyType={self.property_type}" +
               f"&sort={self.sort}" +
               f"&maxPrice={self.maxprice}" +
               f"&minPrice={self.minprice}" +
               f"&language={self.LANGUAGE}" +
               f"&numPage={page}")
        return url

    def search_api(self, page):
        """Perform the API request for a specific page and return the JSON response."""
        url = self.define_search_url(page)
        headers = {'Content-Type': "application/json", 'Authorization': f'Bearer {self.access_token}'}
        response = requests.post(url, headers=headers)
        if response.status_code == 200:
            return response.json()
        else:
            print(f"Request failed with status: {response.status_code}")
            return None

    def results_to_df(self, results):
        """Convert API search results to a DataFrame."""
        return pd.DataFrame.from_dict(results['elementList'])

    def fetch_search_results(self, total_pages, output_filename="idealista_results.csv"):
        """Fetch search results over multiple pages and save to a CSV file."""
        df_tot = pd.DataFrame()

        for page in range(1, total_pages + 1):
            results = self.search_api(page)
            if results:
                df = self.results_to_df(results)
                df_tot = pd.concat([df_tot, df], ignore_index=True)

        # Export to CSV
        df_tot.to_csv(output_filename, index=False)
        print(f"Data successfully saved to {output_filename}")
        return df_tot


# Usage Example:
client = IdealistaAPIClient(
    max_items=50,
    operation='rent',
    property_type='homes',
    order='priceDown',
    locationId='0-EU-ES-28-01-001',
    sort='desc',
    maxprice='1000000000000',
    minprice="0"
)

# Conduct search across multiple pages and get results as CSV
client.fetch_search_results(total_pages=2, output_filename="apartments_for_roomates.csv")

