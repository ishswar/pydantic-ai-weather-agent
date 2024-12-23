import os
import uuid

import pandas as pd
import requests


def calculate_cost(input_tokens, output_tokens):
    """Calculate the total cost based on the number of input and output tokens.
    Assuming model is GPT-4o-mini with the following pricing:
    """
    input_cost_per_million = 0.15  # dollars per million input tokens
    output_cost_per_million = 0.60  # dollars per million output tokens

    input_cost = (input_tokens * input_cost_per_million) / 1_000_000
    output_cost = (output_tokens * output_cost_per_million) / 1_000_000
    total_cost = input_cost + output_cost

    return total_cost

def update_cost_in_csv(conversation_id: uuid, request_tokens: int, response_tokens: int):
    """Update the total cost in the CSV file for a given conversation ID."""
    csv_file = os.environ.get('FUNCTION_CALL_CSV')
    # Check if the CSV file exists
    if os.path.exists(csv_file):
        # Read the CSV file into a DataFrame
        df = pd.read_csv(csv_file)

        # Check if the 'conversation_id' column exists in the DataFrame
        if 'conversation_id' in df.columns:
            # Find the row with the specified conversation_id
            row_index = df[df['conversation_id'] == str(conversation_id)].index

            if len(row_index) > 0:
                # Calculate the cost
                total_cost = calculate_cost(request_tokens, response_tokens)

                # Add the total cost to the row
                df.at[row_index[0], 'total_cost'] = total_cost

                # Write the updated DataFrame back to the CSV file
                df.to_csv(csv_file, index=False)
                print(f"Total cost for {conversation_id}: ${total_cost}")
            else:
                print(f"Conversation ID {conversation_id} not found.")
        else:
            print(f"Conversation ID column not found in CSV file.")
    else:
        print(f"CSV file {csv_file} does not exist.")

def calculate_total_cost_from_csv() -> float:
    """Calculate the total cost from the CSV file."""
    # Retrieve the CSV file path from the environment variable
    csv_file = os.environ.get('FUNCTION_CALL_CSV')

    if not csv_file or not os.path.exists(csv_file):
        # If the file doesn't exist, return 0 as the total cost
        print("CSV file not found. Returning total cost as 0.")
        return 0.0

    # Read the CSV file into a DataFrame
    df = pd.read_csv(csv_file)

    # Check if the required columns exist in the DataFrame
    if 'total_cost' in df.columns :
        # Calculate the sum of the 'total_cost' column
        total_cost = df['total_cost'].sum()
        total_cost = round(total_cost, 5)
        print(f"Total cost: ${total_cost:.4f}")
        return total_cost
    else:
        # If required columns are missing, return 0 as the total cost
        print("Required columns are missing in the CSV file. Returning total cost as 0.")
        return 0.0

def generate_daily_report() -> pd.DataFrame:
    """Generate a daily report from the CSV file."""
    # Retrieve the CSV file path from the environment variable
    csv_file = os.environ.get('FUNCTION_CALL_CSV')

    if not csv_file or not os.path.exists(csv_file):
        # If the file doesn't exist, return an empty DataFrame
        print("CSV file not found. Returning empty report.")
        return pd.DataFrame(columns=["date", "number_of_calls", "total_cost"])

    # Read the CSV file into a DataFrame
    df = pd.read_csv(csv_file)

    # Check if the required columns exist in the DataFrame
    if 'timestamp' in df.columns and 'total_cost' in df.columns:
        # Convert 'timestamp' to datetime
        df['timestamp'] = pd.to_datetime(df['timestamp'])

        # Extract the date from the timestamp
        df['date'] = df['timestamp'].dt.date

        # Group by 'date' and aggregate
        report = df.groupby('date').agg(
            number_of_calls=('conversation_id', 'count'),
            total_cost=('total_cost', 'sum')
        ).reset_index()

        # Optionally, sort the report by date
        report = report.sort_values(by='date')

        return report
    else:
        # If required columns are missing, return an empty DataFrame
        print("Required columns are missing in the CSV file. Returning empty report.")
        return pd.DataFrame(columns=["date", "number_of_calls", "total_cost"])

# Sample prompts for weather queries
sample_prompts = [
            "What's the current temperature in The Big Apple?",
            "How's the weather in The City of Angels today?",
            "Is it windy in The Windy City?",
            "What's the forecast for The City of Light?",
            "Is it snowing in The Eternal City?",
            "How's the weather in The Harbour City?",
            "What's the temperature in The Eastern Capital?",
            "Is it sunny in The Mother City?",
            "What's the weather like in The Big Smoke today?",
            "Is it raining in The Third Rome?",
            "How's the weather in New York City?",
            "Is it foggy in Los Angeles?",
            "What's the forecast for Sydney today?",
            "How's the weather in Cairo?",
            "Is it sunny in Moscow?",
            "What's the temperature in Mumbai, The City of Dreams?",
            "Is it cloudy in Buenos Aires?",
            "What's the forecast for The City of Gold?",
            "How's the weather in Tokyo?",
            "Is it raining in Berlin, The Grey City?",
            "What's the weather like in The Pearl of the Orient?",
            "Is it hot in Rio de Janeiro, The Marvelous City?",
            "What's the temperature in Cape Town, The Mother City?",
            "How's the weather in The City of a Thousand Minarets?",
            "Is it windy in Athens, The Cradle of Western Civilization?",
            "How's the weather in Istanbul, The City on Two Continents?",
            "Is it snowing in The Paris of South America?",
            "How's the weather in Shanghai, The Pearl of the Orient?",
            "Is it sunny in The Marvelous City?",
            "What's the temperature in The City of a Hundred Spires?"
]

# Function to get key information from OpenRouter API
def get_key_info(openrouter_api_key: str) -> str:
    """ Get key information from OpenRouter API """
    OPENAI_API_BASE = os.environ.get("OPENAI_API_BASE")
    if OPENAI_API_BASE:
        # Check if we are using openrouter
        if "openrouter" in OPENAI_API_BASE:
            url = f"{OPENAI_API_BASE}/auth/key"
            headers = {
                "Authorization": f"Bearer {openrouter_api_key}"
            }

            # Perform the GET request
            response = requests.get(url, headers=headers)

            # Check if the request was successful
            if response.status_code == 200:
                data = response.json()

                # Expected structure (type Key):
                # {
                #   "data": {
                #       "label": string,
                #       "usage": number,
                #       "limit": number | null,
                #       "is_free_tier": boolean,
                #       "rate_limit": {
                #           "requests": number,
                #           "interval": string
                #       }
                #   }
                # }

                # Extract values:
                key_data = data["data"]
                label = key_data["label"]
                usage = key_data["usage"]
                limit_val = key_data["limit"]
                is_free_tier = key_data["is_free_tier"]
                rate_limit = key_data["rate_limit"]

                # Format limit to a readable string
                limit_str = str(limit_val) if limit_val is not None else "Unlimited"

                # Build a nice textual summary
                summary = f"""
                OpenRouter Information:  
                  - Usage: {usage:.3f} credits used  
                  - Limit: {limit_str}  
                  - Left: {(limit_val - usage):.3f} credits remaining  
                """
                return summary.strip()
            else:
                print(f"Error: Received status code {response.status_code} from the API")
                return f"Error: Received status code {response.status_code} from the API"
        else:
            return "Non OpenRouter API usage"

