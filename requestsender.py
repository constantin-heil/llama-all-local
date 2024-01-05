import requests
import json
import argparse

ap = argparse.ArgumentParser(
    description = "Send POST requests to the running chat service"
)

ap.add_argument('-i', 
                '--input', 
                help = "Input query text",
                required = True)
ap.add_argument('-f',
                '--full',
                action = 'store_true')

cmd_args = vars(ap.parse_args())

payload = {
    "text": cmd_args['input']
}

response = requests.post(
    url = "http://localhost:5001/chat",
    json = payload
)

parsed_response = json.loads(response.text)

if cmd_args['full']:
    print(f"FULLPROMPT:::{parsed_response['fullprompt']}")

print(f"CODE:{response.status_code}:::{parsed_response['response']}")