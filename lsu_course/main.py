import logging
import os

from dotenv import load_dotenv
from openai import OpenAI
from telegram import Update
from telegram.ext import ApplicationBuilder, CommandHandler, ContextTypes

# Load environment variables from a .env file.
load_dotenv()

# Initialize the OpenAI client with an API key.
openai = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
# Get the Telegram bot token from the environment.
tg_bot_token = os.getenv("TG_BOT_TOKEN")

# A list to store the message history for the OpenAI API.
messages = [
    {"role": "system", "content": "You are a helpful assistant that answers questions."}
]

# Configure the logging level and format.
logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s", level=logging.INFO
)


async def chat(update: Update, context: ContextTypes.DEFAULT_TYPE):
    # Append the user's message to the message history.
    messages.append({"role": "user", "content": update.message.text})
    # Generate a response from the OpenAI API using the accumulated messages.
    completion = openai.chat.completions.create(
        model="gpt-3.5-turbo", messages=messages
    )
    # Extract the response content from the API's response.
    completion_answer = completion.choices[0].message
    # Append the AI's response to the message history.
    messages.append(completion_answer)

    # Send the response back to the user on Telegram.
    await context.bot.send_message(
        chat_id=update.effective_chat.id, text=completion_answer.content
    )


async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    # Initial message when the bot is started.
    await context.bot.send_message(
        chat_id=update.effective_chat.id, text="I'm a bot, please talk to me!"
    )


if __name__ == "__main__":
    # Set up the Telegram bot with the provided token.
    application = ApplicationBuilder().token(tg_bot_token).build()

    # Define command handlers for starting the bot and chatting.
    start_handler = CommandHandler("start", start)
    chat_handler = CommandHandler("chat", chat)

    # Add command handlers to the application.
    application.add_handler(start_handler)
    application.add_handler(chat_handler)

    # Start the bot and poll for new messages.
    application.run_polling()