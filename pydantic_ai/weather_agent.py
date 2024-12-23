from __future__ import annotations as _annotations

import asyncio
import json
import logging
import os
import time
import uuid
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import Any, List, Optional, Literal

import dotenv
import logfire
import pandas as pd
import python_weather
import requests
from devtools import debug
from httpx import AsyncClient
from pydantic import BaseModel
from pydantic import Field
from pydantic_ai import Agent, ModelRetry, RunContext, UserError
from pydantic_ai.messages import UserPromptPart , ModelRequest

import streamlit as st
from pydantic_ai.models.openai import OpenAIModel
from pydantic_ai.settings import ModelSettings, UsageLimits

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

# load env
dotenv.load_dotenv()

OPENAI_API_BASE = os.getenv('OPENAI_API_BASE', 'https://api.openai.com')
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')


# Use openRouter instead of openAI directly

model = OpenAIModel(
    #'mistralai/ministral-8b',  # Non GPT Modules are hit or miss
    "openai/gpt-4o-mini",
    base_url=OPENAI_API_BASE,
    api_key=OPENAI_API_KEY,
)

# The maximum number of requests allowed to the model  on every run of the agent
# Setting resource limits for the agent
usage_limits = UsageLimits(
    request_limit=10,
    request_tokens_limit=8048,
    response_tokens_limit=4048)


# 'if-token-present' means nothing will be sent (and the example will work) if you don't have logfire configured
logfire.configure()

class HourlyForecast(BaseModel):
    time: str  # Formatted time (e.g., "12:00 PM")
    temperature: float  # Temperature in the chosen unit
    description: str  # Weather description
    emoji: str  # Emoji representing the weather kind

class DailyForecast(BaseModel):
    date: str  # Formatted date (e.g., "Monday, December 11, 2024")
    average_temperature: float  # Average temperature for the day
    hourly_forecasts: List[HourlyForecast]  # List of hourly forecasts



class WeatherData(BaseModel):
    city: str = Field(..., description="City for which the weather data is retrieved")
    current_temperature: float = Field(..., description="Current temperature in the chosen unit")
    temp_unit: str = Field(..., description="Temperature unit (C or F)")
    latitude: Optional[float] = Field(None, description="Latitude of the city")
    longitude: Optional[float] = Field(None, description="Longitude of the city")
    # forecasts: List[DailyForecast] = Field(..., description="List of daily forecasts")
    forecasts_id : Optional[str] = Field(None, description="Forecast ID that will be same as Conversation ID")
    country: Optional[str] = Field(None, description="Country of the city")
    region: Optional[str] = Field(None, description="Region of the city")
    unknown_city: Optional[bool] = Field(None, description="True if the city is unknown")
    reason: Optional[str] = Field(None, description="Reason for the unknown city")
    country_emoji: Optional[str] = Field(None, description="Emoji representing the country's flag")
    summary_of_next_x_days: Optional[str] = Field(None, description="Summery of next x days weather")

    def __str__(self) -> str:
        return (f"Weather in {self.city}, {self.country or 'Unknown Country'}: {self.current_temperature}°{self.temp_unit}."
                f"\n{self.summary_of_next_x_days}"
                )

    def __repr__(self) -> str:
        return f"WeatherData(city={self.city!r}, current_temperature={self.current_temperature!r}, temp_unit={self.temp_unit!r}, forecasts_id={self.forecasts_id!r}, country={self.country!r}, region={self.region!r}, unknown_city={self.unknown_city!r}, reason={self.reason!r}, country_emoji={self.country_emoji!r})"

class AgentResult(BaseModel):
    weather_data: Optional[WeatherData] = Field(None, description="Weather data")
    non_weather_related_message: Optional[str] = Field(None, description="Generated in event of non-weather related input")
    quota_key : Optional[str] = Field(None, description="Quota key")

class QuotaStatus(BaseModel):
    quota_reached: bool
    count_today: int = -1 # Default value of -1 indicates that the count is not available
    quota_key: Optional[str] = None

@dataclass
class Deps:
    client: AsyncClient
    geo_api_key: str | None
    weather_agent: Agent[str, Deps] = None
    conversation_id: uuid.UUID = None
    summarize_weather_agent: Agent[str, Deps] = None

@dataclass
class SummarizeData:
    client: AsyncClient
    city_name: str
    country_name: str | None
    conversation_id: uuid.UUID = None



input_checker_agent = Agent(
    model,
    system_prompt=(
        "First, use the 'quota_checker' tool to verify if the quota has been exceeded. "
        "If the quota is exceeded, stop the process immediately. "
        "If the quota is not exceeded, proceed to assess whether the user's question pertains to weather. "
        "If the question is unrelated to weather, kindly ask the user to provide a weather-related question. "
        "If the question includes a city name or other details, assess its relevance to weather and respond accordingly. "
        "For weather-related inquiries, utilize available tools to retrieve accurate weather data."
        "You never generate weather data; you only fetch it from the provided tools."
    ),
    deps_type=Deps,
    retries=2,
    result_type=AgentResult,

)


@input_checker_agent.tool
def quota_checker(ctx: RunContext[Deps], thoughts: str, user_input: str) -> QuotaStatus | RuntimeError:
    """Check if we have quota left or exceeded the quota for the day."""
    # print agent thoughts
    logger.info(f"[Tool quota_checker] Thoughts: {thoughts}")
    # Define the CSV file path
    csv_file = os.environ.get('FUNCTION_CALL_CSV', 'function_calls.csv')

    # Get the current date
    today = datetime.now().date()

    # Check if the CSV file exists
    if os.path.exists(csv_file):
        logger.info(f"CSV file exists")
        # Read the CSV file into a DataFrame
        df = pd.read_csv(csv_file)
        # Convert the 'timestamp' column to datetime
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        # Filter for today's entries
        today_df = df[df['timestamp'].dt.date == today]
        # Count the number of entries for today
        count_today = today_df.shape[0]
    else:
        logger.info(f"CSV file does not exists")
        # If the file doesn't exist, initialize count_today to 0
        count_today = 0
        # Create a new DataFrame with the appropriate columns
        df = pd.DataFrame(columns=['timestamp', 'user_input'])

    # Define the threshold
    threshold = os.environ.get('MAX_QUOTA', 10)

    if count_today < int(threshold):
        # Append the new entry
        new_entry = pd.DataFrame({
            'timestamp': [datetime.now()],
            'user_input': [user_input],
            'conversation_id': [ctx.deps.conversation_id]
        })
        df = pd.concat([df, new_entry], ignore_index=True)
        # Write the updated DataFrame back to the CSV file
        df.to_csv(csv_file, index=False)
        logger.info(f"Quota not reached. Count: {count_today}")
        return QuotaStatus(quota_reached=False, count_today=count_today, quota_key='qxzy34fg')
    else:
        logger.info(f"Quota reached. Count: {count_today}")
        raise RuntimeError('Max quota reached for the day. Please try again tomorrow.')


@input_checker_agent.tool
async def check_weather(ctx: RunContext[Deps], thoughts: str, user_input: str ) -> Deps:
    """Use weather agent to get the weather data."""

    deps = Deps(
        client= ctx.deps.client , geo_api_key=ctx.deps.geo_api_key,
        weather_agent=ctx.deps.weather_agent, conversation_id=ctx.deps.conversation_id,
        summarize_weather_agent=ctx.deps.summarize_weather_agent
    )
    print(f"[Tool check_weather] Thoughts [{thoughts}] and User Input [{user_input}]")

    result = await ctx.deps.weather_agent.run(user_input, deps=deps, usage_limits=usage_limits)

    return result.data

@input_checker_agent.result_validator
async def check_response_object(reply: AgentResult) ->  AgentResult:
    """Check if the response object is valid."""
    logger.info(f"Checking response object")
    print("Checking response object")
    debug(reply)
    if reply.weather_data and not reply.non_weather_related_message:
        forecast_id = reply.weather_data.forecasts_id
        if not forecast_id:
            logger.error('Forecast ID is missing')
            raise ModelRetry('Forecast ID is missing - make sure to populate the forecast_id field')

        if reply.quota_key == "qxzy34fg":
            logger.info(f"Quota key: {reply.quota_key}")
        else:
            logger.error(f"Quota key not found")
            raise ModelRetry('Quota key is not what was expected')


    return reply


summarize_weather_agent = Agent(
    model,
    system_prompt=(
        "Summarize the weather data for the next x days. keep it brief and concise. And do try to answer the user's question directly."
        "For example, if the user asks 'What's the weather like in Buffalo?', summarize the weather for the next x days in the reply"
        "You will be provided forcast data as json - use only that data to summarize the weather for next x days"
        "IMPORTANT: Don't generate weather data; only summarize the provided forecast data."
    ),
    deps_type=SummarizeData,
    retries=2,
    result_type=str
)

@summarize_weather_agent.system_prompt
async def summarize_weather(ctx: RunContext[SummarizeData]
                            ) -> str:
    # print(f"[System Prompt] Thoughts: {thoughts}")

    conversation_id = str(ctx.deps.conversation_id)
    forecast_json = ""
    if conversation_id:
        if conversation_id + "_data" in st.session_state:
            daily_forecasts = st.session_state[conversation_id + "_data"]
            forecast_json = json.dumps([item.model_dump_json() for item in daily_forecasts])
            # print(f"Forecast data: {forecast_json}")
            return f"""Summarizing the weather forecast for {ctx.deps.city_name} for the next 3 days.
                Forecast data: {forecast_json}"""
        else:
            raise ModelRetry('Forecast data not found in session state')
    else:
        raise ModelRetry('Conversation ID not found')



weather_agent = Agent(
    model,
system_prompt=(
    "Be concise and respond in one sentence for 'summary_of_next_x_days'. Ensure that you directly address the user's question in the summary. Only once you have weather data and forecast id - at last Use tool 'summarize_forecast' to generate the summary. "
    "For example: if the user asks 'Is it cloudy in Sydney?', and it is or will be cloudy, respond with 'Yes, it's cloudy'; otherwise, say 'No'. "
    "Similarly, if the user asks 'Do I need sunglasses?' and it’s not sunny, say 'No, you don't need sunglasses', or if it’s going to rain, respond with 'Yes, it’s going to rain, please carry an umbrella.' "
    "First, verify if the user's question pertains to weather. If it doesn't, ask them to provide a weather-related question. "
    "For valid weather inquiries, identify the city or country. If the location is invalid, ask for a valid one. "
    "If a country is specified, include its flag emoji in the response under 'country_emoji'. "
    "Handle potential typos in city names by identifying the correct city. "
    "Always use the appropriate temperature unit (Celsius or Fahrenheit) based on the country or region. "
    "If a state or country is provided, find its capital city and provide weather information for that city. "
    "IMPORTANT: If the user is asking for the weather in a specific city or country, summarize the weather (after you have forecast id)  for the next x days in the 'summary_of_next_x_days' field using tool 'summarize_forecast'."
    "If the user provides just a city name without a specific question, summarize the weather for the next x days in 'summary_of_next_x_days' as a short 2-3 line description of the weather for that location. "
    "NEVER use your own knowledge of the world weather data as it might be outdated, and it was collected during your training. "
    "Always use the provided tools to fetch the latest weather data. DO NOT rely on your internal knowledge for weather-related responses; always use the tools provided to fetch the most up-to-date information."
    )
    ,
    deps_type=Deps,
    retries=2,
    result_type=WeatherData,
)



class Kind(Enum):
    CLOUDY = '☁️'
    FOG = '💨'
    HEAVY_RAIN = '🌧️'
    HEAVY_SHOWERS = '🌦️'
    HEAVY_SNOW = '❄️'
    HEAVY_SNOW_SHOWERS = '🌨️'
    LIGHT_RAIN = '🌦️'
    LIGHT_SHOWERS = '🌦️'
    LIGHT_SLEET = '🌧️'
    LIGHT_SLEET_SHOWERS = '🌧️'
    LIGHT_SNOW = '🌨️'
    LIGHT_SNOW_SHOWERS = '🌨️'
    PARTLY_CLOUDY = '⛅'
    SUNNY = '☀️'
    THUNDERY_HEAVY_RAIN = '⛈️'
    THUNDERY_SHOWERS = '⛈️'
    THUNDERY_SNOW_SHOWERS = '❄️'
    VERY_CLOUDY = '☁️'

async def get_weather_call(city_name: str, country_name: str | None, temp_unit_c_or_f: str):
    """Get the weather information for a given city."""
    if temp_unit_c_or_f == 'C':
        temp_unit = python_weather.METRIC
        # Print what the temperature unit agent is using
        logger.info(f"Temperature unit: Celsius")
    else:
        temp_unit = python_weather.IMPERIAL
        # Print what the temperature unit agent is using

    # Declare the client for fetching weather data
    async with python_weather.Client(unit=temp_unit) as client:
        try:
            weather = await client.get(city_name+', '+country_name)
            daily_forecasts = []

            # Process daily forecasts
            for daily in weather:  # Limit to the next 3 days
                hourly_forecasts = []
                for hourly in daily:
                    kind_emoji = Kind[hourly.kind.name].value if hourly.kind.name in Kind.__members__ else ''
                    hourly_forecasts.append(HourlyForecast(
                        time=hourly.time.strftime('%I:%M %p'),
                        temperature=hourly.temperature,
                        description=hourly.description,
                        emoji=kind_emoji
                    ))

                daily_forecasts.append(DailyForecast(
                    date=daily.date.strftime('%A, %B %d, %Y'),
                    average_temperature=daily.temperature,
                    hourly_forecasts=hourly_forecasts
                ))
        except Exception as e:
            print(e)
            daily_forecasts = []
            weather = None

        return daily_forecasts, weather


@weather_agent.tool
async def summarize_forecast(ctx: RunContext[Deps], thoughts: str , city_name: str ,
                             forecasts_id : str , prompt: str, country_name: str ) -> str:
    """Summarize the forecast for the next x days."""

    if not forecasts_id:
        raise ModelRetry('Forecast ID is required')

    summarize_data = SummarizeData(client=ctx.deps.client, city_name=city_name,
                                   country_name=country_name,
                                   conversation_id=ctx.deps.conversation_id
                                   )

    user_prompt = prompt  # Default to empty string

    # Check if messages[0] exists and has the expected structure
    if ctx.messages and isinstance(ctx.messages[0],
                                   ModelRequest):  # Assuming ModelRequest is the correct type for messages
        message = ctx.messages[0]

        # Check if parts exist and there are at least 2 parts
        if hasattr(message, 'parts') and len(message.parts) > 1:
            # Check if the second part is of type UserPromptPart
            if isinstance(message.parts[1], UserPromptPart):
                user_prompt = message.parts[1].content
            else:
                print("Error: The second part is not of type UserPromptPart.")
        else:
            print("Error: The message does not have enough parts or 'parts' is missing.")
    else:
        print("Error: The message structure is not as expected.")

    print("User prompt:", user_prompt)


    summary = await ctx.deps.summarize_weather_agent.run(user_prompt, deps=summarize_data, usage_limits=usage_limits)

    # summary = f"Summarizing the weather forecast for {city_name} for the next 3 days."

    return summary.data

@weather_agent.tool
async def get_lat_lng(ctx: RunContext[Deps], thoughts: str , city_name: str , iso_3166_alpha_2_country_name: str | None) -> dict[str, Any]:
    """Get the latitude and longitude of a given city."""
    # print agent thoughts
    logger.info(f"[Tool quota_checker] Thoughts: {thoughts}")
    logger.info(f"Getting lat and lng for {city_name} and Country {iso_3166_alpha_2_country_name}")
    try:
        api_key = ctx.deps.geo_api_key
        if not api_key:
            # User has not provided the API key - so return 0,0 , Now it's up to LLM to handle this
            return {'lat' : 0, 'lang': 0}
        if not city_name:
            raise ModelRetry('City name is required')
        if not iso_3166_alpha_2_country_name:
            raise ModelRetry('Country name is required')


        logger.info(f"Getting lat and lng for {city_name} and Country {iso_3166_alpha_2_country_name}")
        url = f'https://api.mapbox.com/geocoding/v5/mapbox.places/{city_name}.json?country={iso_3166_alpha_2_country_name}&access_token={api_key}'
        response = requests.get(url).json()
        if response['features']:
            lat = response['features'][0]['center'][1]
            lng = response['features'][0]['center'][0]
            # return the lat and lng using logger
            logger.info(f"Latitude: {lat}, Longitude: {lng}")
            return {'lat' : lat, 'lang': lng}
    except Exception as e:
        logger.error(f'Error getting lat and lng: {e}', exc_info=True)
        raise ModelRetry('Could not find the location')

    raise ModelRetry('Could not find the location')



@weather_agent.tool
async def getweather(
    ctx: RunContext[Deps],
    city_name: str,
    country_name: str | None,
    temperature_unit_celsius_or_fahrenheit: Literal['C', 'F'],
    latitude: str,
    longitude: str,
    thoughts: str
) -> WeatherData:
    """Get the weather data for a given city."""
    logger.info(f"[Tool : getweather ] Thoughts: {thoughts}")
    # Await the cached result of get_weather_call
    daily_forecasts, weather = await get_weather_call(city_name, country_name, temperature_unit_celsius_or_fahrenheit)

    if daily_forecasts == [] and weather is None:
        logger.error('Could not find the daily forecasts and weather data')
        raise ModelRetry('Could not find the daily forecasts and weather data')

    if len(daily_forecasts) == 0:
        logger.error('Could not find the daily forecasts')
        raise ModelRetry('Could not find the daily forecasts')
    else:
        date = daily_forecasts[0].date
        # Get the current date and time
        current_datetime = datetime.now()
        # Extract the year
        current_year = current_datetime.year
        if str(current_year) not in str(date):
            logger.error('Could not find the current year in the date')
            raise ModelRetry('Could not find the current year in the date')

    forecast_id = '---'
    if ctx.deps.conversation_id:
        forecast_id = str(ctx.deps.conversation_id)
        if forecast_id + "_data" not in st.session_state:
            st.session_state[forecast_id + "_data"] = daily_forecasts
            logger.info(f"Stored forecast data in session state with key {forecast_id + '_data'}")
    forecast_json = [item.model_dump_json() for item in daily_forecasts]

    ret_data = WeatherData(
        city=city_name,
        # forecasts=daily_forecasts,
        forecasts_id=forecast_id,
        current_temperature=weather.temperature,
        temp_unit=temperature_unit_celsius_or_fahrenheit,
        country=weather.country,
        region=weather.region,
        latitude=float(latitude) if latitude != '' else None,
        longitude=float(longitude) if longitude != '' else None,
    )
    #    debug(ret_data)
    return ret_data


# Standalone function to get the weather data for a given city
async def main():
    async with AsyncClient() as client:
        dotenv.load_dotenv()
        geo_api_key = os.getenv('WEATHER_MAP_BOX_API_KEY')
        conversation_id = uuid.uuid4()
        deps = Deps(
            client=client, geo_api_key=geo_api_key,
            weather_agent=weather_agent,
            conversation_id=conversation_id
        )
        start_time = time.time()
        result = await input_checker_agent.run(
            'What is the weather like in Buffalo?', deps=deps, usage_limits=usage_limits
        )
        end_time = time.time()
        print(f"Time taken: {end_time - start_time:.2f} seconds")
        # debug(result)
        print('Response:', result.data.model_dump_json(indent=2))

        print('Cost:', result.cost)
        # debug(result.all_messages())

# Will be only call if this file is run directly
if __name__ == '__main__':
    asyncio.run(main())