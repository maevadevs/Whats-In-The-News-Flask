# Example for calling the endpoint of the API
import requests
import json

# Define the URL of the API
# Flask uses Port 5000 by default
api_url = 'http://0.0.0.0:5000/api/'

# Define the dat we want to pass to the API
data = [[14.34, 1.68, 2.7, 25.0, 98.0, 2.8, 1.31, 0.53, 2.7, 13.0, 0.57, 1.96, 660.0]]

# Convert the data into JSON to pass over
json_data = json.dumps(data)

# Define header for the request call
headers = {
  "content-type": "application/json",
  "Accept-Charset": "UTF-8"
}

# Build a request to send
result = requests.post(
  api_url, 
  data=json_data, 
  headers=headers
)

# Check the result
print(result, result.text)
