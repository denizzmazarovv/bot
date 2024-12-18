from aiogram import Bot, Dispatcher, types, F
from aiogram.filters import Command
from transformers import AutoModelForCausalLM, AutoTokenizer
# from config import TELEGRAM_TOKEN  # Импорт токена из config.py
import os
TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN")
import logging  # Стандартный модуль logging
import asyncio  # Для запуска асинхронного main

# Настройка логирования
logging.basicConfig(level=logging.INFO)

# Инициализация бота
bot = Bot(token=TELEGRAM_TOKEN)

# Инициализация диспетчера
dp = Dispatcher()

# Загрузка модели DistilGPT-2
model_name = "distilgpt2"
model = AutoModelForCausalLM.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Функция для генерации ответа
async def generate_response(user_input: str):
    inputs = tokenizer(user_input, return_tensors="pt")
    outputs = model.generate(**inputs, max_length=30)
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

# Обработчик команды /start
@dp.message(Command("start"))
async def send_welcome(message: types.Message):
    await message.reply("Привет! Напиши что-нибудь, и я сгенерирую ответ!")

# Обработчик текстовых сообщений
@dp.message(F.text)  # Проверка, что пришёл текст
async def echo(message: types.Message):
    user_input = message.text
    response = await generate_response(user_input)
    await message.reply(response, parse_mode="Markdown")

# Основная функция запуска бота
async def main():
    # Запуск polling с обработкой обновлений
    await dp.start_polling(bot, skip_updates=True)

# Запуск бота
if __name__ == '__main__':
    asyncio.run(main())