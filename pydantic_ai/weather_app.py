import copy
import io
import random
import sys
import time

import dotenv
import pydantic_ai
from devtools import debug
import pandas as pd
from dotenv import load_dotenv
from httpx import AsyncClient
from datetime import datetime
import streamlit as st
import asyncio

import os
import uuid
from pydantic import BaseModel
from pydantic_ai.messages import ModelTextResponse, UserPrompt
from pydantic_ai.result import ResultData

from utils.all_utils import calculate_cost, update_cost_in_csv, calculate_total_cost_from_csv, generate_daily_report, \
    sample_prompts
from weather_agent import WeatherData, input_checker_agent
# from web_search_agent import web_search_agent, Deps
from weather_agent import weather_agent, Deps
import logging
import plotly.graph_objects as go

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

class ConversationCostData(BaseModel):
    request_tokens: int
    response_tokens: int
    dollar_cost: float


load_dotenv()


st.set_page_config(page_title="Pydantic AI Weather Chatbot", page_icon="🌦️")

chat_tab, sample_prompts_tab, cost_tab = st.tabs(["Chat", "Sample prompts","Cost Analysis"])


# @st.cache_resource
async def prompt_ai(messages) -> tuple[ResultData, uuid.UUID]:
    async with AsyncClient() as client:
        conversation_id = uuid.uuid4()
        geo_api_key = os.getenv('WEATHER_MAP_BOX_API_KEY')
        deps = Deps(
            client=client, geo_api_key=geo_api_key,
            weather_agent=weather_agent,
            conversation_id=conversation_id
        )

        for message in messages:
            if isinstance(message, ModelTextResponse):
                # Call the __str__ method on WeatherData inside ModelTextResponse
                message.content = str(message.content)  # This calls __str__ on WeatherData

        # All the magic happens here
        # Call the agent with the last message in the conversation ; this is what user has asked
        result = await input_checker_agent.run(
            messages[-1].content, deps=deps
        )



        return result , conversation_id

            # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

def print_weather_data(weather):
    uuid_ = uuid.uuid4()

    region_str = weather.region if weather.region else ""
    if region_str:
        region_str = f"{region_str} -"
    # Assuming 'weather' is an object with the necessary attributes
    st.markdown(
        f"**Current Temperature in {weather.city}** [{region_str} {weather.country}]: "
        f"<span style='color: orange; font-size: 24px;'>{weather.current_temperature}° {weather.temp_unit}</span>",
        unsafe_allow_html=True
    )
    st.write(f"{weather.summary_of_next_x_days}")
    # debug(weather)
    latitude = weather.latitude
    longitude = weather.longitude
    # st.write(f"latitude: {latitude}, longitude: {longitude}")
    if latitude is not None and longitude is not None:

        fig = go.Figure(go.Scattermap(
            lat=[latitude],
            lon=[longitude],
            mode='markers',
            marker=go.scattermap.Marker(
                size=14,
            ),
            text=[f"{weather.city}: {weather.current_temperature}° {weather.temp_unit}"],
        ))

        fig.update_layout(
            hovermode='closest',
            map=dict(
                bearing=0,
                center=go.layout.map.Center(
                    lat=latitude,
                    lon=longitude
                ),
                pitch=0,
                zoom=10
            ),
            annotations=[
                go.layout.Annotation(
                    x=0.02,
                    y=1,
                    xref='paper',
                    yref='paper',
                    text=weather.country_emoji,
                    showarrow=False,
                    font=dict(size=55),
                    align='left',
                    valign='top'
                )
            ]
            ,
            margin=dict(t=0, b=0, l=0, r=0),  # Remove padding around the plot area
        )
        st.plotly_chart(fig, key=f"map-{uuid_}")

    st.subheader(f"Weather Forecast for the next {len(weather.forecasts)} days:", divider=True)
    for daily in weather.forecasts:

        st.subheader(f"{daily.date}")
        # Create DataFrame from hourly forecasts
        df = pd.DataFrame([
            {
                'Hour': hourly.time,
                'Temperature': hourly.temperature,
                'Description': hourly.emoji + "  " + hourly.description,
                'emoji': hourly.emoji
            }
            for hourly in daily.hourly_forecasts
        ])
        most_frequent_emoji = ''
        most_frequent_data = None
        if len(df) > 0:
            # Count the occurrences of each description
            description_counts = df['Description'].value_counts()

            # Get the most frequent description
            most_frequent_description = description_counts.idxmax()

            # Filter the DataFrame to include only rows with the most frequent description
            most_frequent_data = df[df['Description'] == most_frequent_description]

            # Get the corresponding emoji
            most_frequent_emoji = most_frequent_data['emoji'].iloc[0]

        st.metric(label="Average Temperature",
                  value=f"{daily.average_temperature} °{weather.temp_unit} {most_frequent_emoji}" ,
                  help=f"{most_frequent_data['Description'].iloc[0] if most_frequent_data is not None else 'Weather Description'}")


        if len(df) > 0:
            st.write("Hourly Temperature Data:")

            tale_area, chart_area = st.columns([6, 4])

            # Display the DataFrame with hourly temperature data
            with tale_area:
                # Add a blank space to the DataFrame so chart and table are aligned
                st.html("&nbsp")
                st.dataframe(df, hide_index=True,
                             column_config={
                                 'emoji': None, # Hide the emoji column
                             })
            with chart_area:

                fig = go.Figure()

                # Add the line chart with emojis
                fig.add_trace(go.Scatter(
                    x=df['Hour'],  # Use the x_values column from the DataFrame
                    y=df['Temperature'],  # Use the y_values column from the DataFrame
                    mode='lines+markers+text',
                    name='Line with emojis',
                    text=df['emoji'],  # Adding emojis from the DataFrame
                    textposition='top center',  # Position of the text
                    marker=dict(size=5, color='blue'),  # Custom marker properties
                    hovertext= df['Hour'] + '<br> ' + df['Description'],  # Hover text to display word descriptions from DataFrame
                    hoverinfo='text',  # Only show the hover text on hover
                    textfont=dict(size=25),  # Adjust the font size of the text (emojis)
                    line_shape='spline',  # Use a spline line shape
                ))

                # Force the x-axis to maintain the order in the dataframe
                fig.update_xaxes(type='category', categoryorder='array', categoryarray=df['Hour'])
                fig.update_layout(
                    margin=dict(t=50),  # Remove padding around the plot area
                    xaxis=dict(
                        title="Hour of the Day",  # X-axis label
                        tickangle=45  # Rotate x-axis labels to 45 degrees
                    ),
                    yaxis=dict(
                        title=f"Temperature (°{weather.temp_unit})",  # Y-axis label
                    ),
                )
                uuid_ = uuid.uuid4() # Each plotly chart needs a unique key
                st.plotly_chart(fig, use_container_width=True, key=f"hourly-temperature-{uuid_}")


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# ~~~~~~~~~~~~~~~~~~ Main Function with UI Creation ~~~~~~~~~~~~~~~~~~~~
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

async def main():

    with chat_tab:
        st.image(image="https://ai.pydantic.dev/img/logo-white.svg")
        st.title("Pydantic AI Weather Chatbot")
        st.caption(f"Powered by multi agent Pydantic AI [v{pydantic_ai.__version__}]")
        # Apply custom CSS to fix the chat input at the bottom
        st.markdown("""
            <style>
                .stChatInput {
                    position: fixed;
                    bottom: 0;
                    border-top: 1px solid #ddd;
                    z-index: 9999;
                }
            </style>
        """, unsafe_allow_html=True)

        # Start Show only code - show image on first load
        # Check if the 'image_s ≈hown' flag exists in session_state
        if 'image_shown' not in st.session_state:
            st.session_state.image_shown = False

        # Show the image on the first load
        if not st.session_state.image_shown:
            # Display the image
            st.image("static/weather_bot.png")
            # Select 4 random prompts from the list
            random_prompts = random.sample(sample_prompts, 4)

            # Create a container with a border
                # Display each random prompt as a bullet point
            st.subheader(f"Here are some prompts to get you started:", divider=True)
            # Create two columns
            col1, col2 = st.columns(2)

            # Display the first two prompts in the first column
            with col1:
                with st.container(border=True):
                    st.write(f"{random_prompts[0]}")
                with st.container(border=True):
                    st.write(f"{random_prompts[1]}")

            # Display the last two prompts in the second column
            with col2:
                with st.container(border=True):
                    st.write(f"{random_prompts[2]}")
                with st.container(border=True):
                    st.write(f"{random_prompts[3]}")

            # use point down emoji to show user to type question in chat
            st.write("👇 Type your weather related questions in the chat below 👇")
            # Set the flag to True to indicate the image has been shown
            st.session_state.image_shown = True
        # END of Show only code - show image on first load

        # Initialize chat history
        if "messages" not in st.session_state:
            st.session_state.messages = []

            # Display chat messages from history on app rerun
        for message in st.session_state.messages:
            role = message.role
            if role in ["user", "model-text-response"]:
                with st.chat_message("human" if role == "user" else "ai" , avatar="🧑‍💻" if role == "user" else "🤖"):
                    if isinstance(message.content, WeatherData):
                        print_weather_data(message.content)
                    else:
                        st.markdown(message.content)

        # React to user input
        if prompt := st.chat_input("What is weather like in New York?"):
            # Display user message in chat message container
            st.chat_message("user" , avatar="🧑‍💻").markdown(prompt)
            # Add user message to chat history
            st.session_state.messages.append(UserPrompt(content=prompt))

            # Display assistant response in chat message container
            response_content = ""
            with st.chat_message("assistant", avatar="🤖"):
                try:
                    messages_copy = copy.deepcopy(st.session_state.messages)
                    with st.spinner("Fetching weather data..."):
                        logger.info(f"Calling agent with messages: {messages_copy}")
                        start_time = time.time()
                        agent_result, conversation_id = await prompt_ai(messages_copy)
                        weather = agent_result.data # Get the weather data from the agent result
                        end_time = time.time()
                        logger.info(f"Time taken to fetch weather data: {(end_time - start_time)}")
                except Exception as e:
                    st.error(f"An error occurred: {str(e)}")
                    response_content = f"An error occurred while calling agent : {str(e)}"
                    print(f"An error occurred while calling agent : {str(e)}")
                    logger.error(f"An error occurred while calling agent : {str(e)}", exc_info=True)
                    st.session_state.messages.append(ModelTextResponse(content=response_content))
                    return

                # Start cost calculation work
                cost_object = agent_result.cost()  # Call the function to get the object
                total_cost = calculate_cost(cost_object.request_tokens, cost_object.response_tokens)

                update_cost_in_csv(conversation_id, cost_object.request_tokens, cost_object.response_tokens)
                cost_data = ConversationCostData(
                    request_tokens=cost_object.request_tokens,
                    response_tokens=cost_object.response_tokens,
                    dollar_cost=total_cost
                )
                # End cost calculation work

                if isinstance(weather.weather_data, WeatherData):
                    if weather.weather_data.unknown_city:
                        response_content = f"Sorry, I couldn't find any information [{weather.reason}]"
                        st.error(response_content)
                    else:
                        if isinstance(weather.weather_data, WeatherData):
                            if len(weather.weather_data.forecasts) == 0:
                                response_content = f"Sorry, I couldn't find any information - Check the city name and try again."
                                st.error(response_content)
                            else:
                                response_content = weather.weather_data

                                print_weather_data(weather.weather_data)
                                st.divider()
                                st.caption(f"Total cost for this conversation: ${cost_data.dollar_cost:.4f},"
                                           f" Request Tokens: {cost_data.request_tokens},"
                                           f" Response Tokens: {cost_data.response_tokens}")
                                st.caption(f"Total time taken to fetch weather data: {round(end_time - start_time)} seconds")
                                # Start of code to capture agent conversation history
                                # Create a StringIO object to capture the output
                                output_buffer = io.StringIO()

                                # Backup the original stdout
                                original_stdout = sys.stdout

                                # Redirect stdout to the StringIO buffer
                                sys.stdout = output_buffer

                                debug(agent_result.all_messages())

                                # Reset stdout to its original value
                                sys.stdout = original_stdout

                                # Get the captured output as a string
                                captured_output = output_buffer.getvalue()

                                st.expander("Click to see agent conversation history", expanded=False).code(captured_output)

                                # Don't forget to close the buffer when you're done
                                output_buffer.close()
                                # End of code to capture agent conversation history
                else:
                    # If the response is not WeatherData, display the response content
                    response_content = weather.non_weather_related_message
                    st.write(response_content)

            st.session_state.messages.append(ModelTextResponse(content=response_content))
    with sample_prompts_tab:
        st.title("Sample Prompts")
        st.write("This section displays some sample prompts to try out the chatbot.")
        for prompt in sample_prompts:
            st.write(f"- {prompt}")
    with cost_tab:
        st.title("Cost Analysis")
        st.write("This section displays the cost analysis for the conversations.")
        total_cost = calculate_total_cost_from_csv()
        st.metric(label="Total Cost", value=f"${total_cost}")
        daily_cost_df = generate_daily_report()
        st.write("Daily Cost Report:")
        st.dataframe(daily_cost_df, hide_index=True)
        daily_quota = os.getenv('MAX_QUOTA')
        st.caption(f"Daily Quota: {daily_quota}")


if __name__ == "__main__":
    asyncio.run(main())