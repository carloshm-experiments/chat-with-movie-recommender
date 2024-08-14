import json
import ast
import os
from openai import AsyncAzureOpenAI
import chainlit as cl
import pickle
import pandas as pd


cl.instrument_openai()

client = AsyncAzureOpenAI()

MAX_ITER = 5

def recommend(movie):
    movie_index = movies[movies['title'] == movie].index[0]
    distances = similarity[movie_index]
    movies_list = sorted(list(enumerate(distances)), reverse=True, key=lambda x: x[1])[1:6]

    recommended_movies = []

    for i in movies_list:
        movie_id = movies.iloc[i[0]].movie_id
        recommended_movies.append(movies.iloc[i[0]].title)
    return recommended_movies

movies_dict = pickle.load(open('./recommender/movies_dict.pkl', 'rb'))
movies = pd.DataFrame(movies_dict)
similarity = pickle.load(open('./recommender/similarity.pkl', 'rb'))

def get_movies_recommendation(movie):
    movie_names = recommend(movie)

    return json.dumps(movie_names)

def get_current_weather(location, unit):
    unit = unit or "Fahrenheit"
    weather_info = {
        "location": location,
        "temperature": "60",
        "unit": unit,
        "forecast": ["windy"],
    }
    return json.dumps(weather_info)

tools = [
    {
        "type": "function",
        "function": {
            "name": "get_current_weather",
            "description": "Get the current weather in a given location",
            "parameters": {
                "type": "object",
                "properties": {
                    "location": {
                        "type": "string",
                        "description": "The city and state, e.g. San Francisco, CA",
                    },
                    "unit": {"type": "string", "enum": ["celsius", "fahrenheit"]},
                },
                "required": ["location"],
            },
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_movies_recommendation",
            "description": "Get a list of recommended movies for a given movie",
            "parameters": {
                "type": "object",
                "properties": {
                    "movie": {
                        "type": "string",
                        "description": "The movie the user wants a recommendation from, e.g. Avatar",
                    }
                },
                "required": ["movie"],
            },
        }
    }
]


@cl.on_chat_start
def start_chat():
    cl.user_session.set(
        "message_history",
        [{"role": "system", "content": "You are a helpful assistant."}],
    )


@cl.step(type="tool")
async def call_tool(tool_call_id, name, arguments, message_history):
    arguments = ast.literal_eval(arguments)

    current_step = cl.context.current_step
    current_step.name = name
    current_step.input = arguments

    if name == "get_current_weather":
        function_response = get_current_weather(
            location=arguments.get("location"),
            unit=arguments.get("unit"),
        )
    elif name == "get_movies_recommendation":
        function_response = get_movies_recommendation(
            movie=arguments.get("movie")
        )

    current_step.output = function_response
    current_step.language = "json"

    message_history.append(
        {
            "role": "function",
            "name": name,
            "content": function_response,
            "tool_call_id": tool_call_id,
        }
    )

async def call_gpt4(message_history):
    settings = {
        "model": "gpt4-8",
        "tools": tools,
        "tool_choice": "auto",
        "temperature": 0,
        "max_tokens": 500,
        "top_p": 1,
        "frequency_penalty": 0,
        "presence_penalty": 0,
    }

    stream = await client.chat.completions.create(
        messages=message_history, stream=True, **settings
    )

    tool_call_id = None
    function_output = {"name": "", "arguments": ""}

    final_answer = cl.Message(content="", author="Answer")

    async for part in stream:
        try:
            new_delta = part.choices[0].delta
            tool_call = new_delta.tool_calls and new_delta.tool_calls[0]
            function = tool_call and tool_call.function
            if tool_call and tool_call.id:
                tool_call_id = tool_call.id

            if function:
                if function.name:
                    function_output["name"] = function.name
                else:
                    function_output["arguments"] += function.arguments
            if new_delta.content:
                if not final_answer.content:
                    await final_answer.send()
                await final_answer.stream_token(new_delta.content)
        except Exception as e:
            print(f"An error occurred: {e}")

    if tool_call_id:
        await call_tool(
            tool_call_id,
            function_output["name"],
            function_output["arguments"],
            message_history,
        )

    if final_answer.content:
        await final_answer.update()

    return tool_call_id


@cl.on_message
async def on_message(message: cl.Message):
    message_history = cl.user_session.get("message_history")
    message_history.append({"role": "user", "content": message.content})

    cur_iter = 0

    while cur_iter < MAX_ITER:
        tool_call_id = await call_gpt4(message_history)
        if not tool_call_id:
            break

        cur_iter += 1