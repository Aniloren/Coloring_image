import telebot
from utils import make_colore
TOKEN = "1740843749:AAGbJ_qaLqp3UjXtAxxejHgNxsmsst-SO5o"
bot = telebot.TeleBot(TOKEN)


@bot.message_handler(commands=['start', 'help'])
def send_welcome(message):
    bot.reply_to(
        message, f'Я красильщик. Приятно познакомиться, {message.from_user.first_name}')

# def start_message(message):

    #keyboard = telebot.types.ReplyKeyboardMarkup(True)
    #keyboard.row('1button', '2button')
    #bot.send_message(message.chat.id, 'Привет!', reply_markup=keyboard)

# @bot.message_handler(content_types=['text'])
# def get_text_messages(message):
#     if message.text == 'Привет!':
#         bot.send_message(message.from_user.id, 'Привет')
#     else:
#         bot.send_message(message.from_user.id, '.')


@bot.message_handler(content_types=["document"])
def get_image(message):
    fileID = message.document.file_id
    file_info = bot.get_file(message.document.file_id)
    downloaded_file = bot.download_file(file_info.file_path)

    filename = "image.jpg"

    with open(filename, 'wb') as new_file:
        new_file.write(downloaded_file)

    filename2 = make_colore(filename)

    with open(filename2, "rb") as file:
        fake_picture = file.read()

    bot.send_photo(message.from_user.id, photo=fake_picture)


@bot.message_handler(content_types=['photo'])
def get_image(message):
    fileID = message.photo[-1].file_id
    file_info = bot.get_file(fileID)
    downloaded_file = bot.download_file(file_info.file_path)

    filename = "get_image.jpg"

    with open(filename, 'wb') as new_file:
        new_file.write(downloaded_file)

    filename2 = make_colore(filename)
    with open(filename2, "rb") as file:
        fake_picture = file.read()

    bot.send_photo(message.from_user.id, photo=fake_picture)


# @bot.message_handler(commands=['switch'])
# def switch(message):
#     markup = bot.types.InlineKeyboardMarkup()
#     switch_button = bot.types.InlineKeyboardButton(text='Try', switch_inline_query="Telegram")
#     markup.add(switch_button)
#     bot.send_message(message.chat.id, "Выбрать чат", reply_markup=markup)


bot.polling(none_stop=True)
