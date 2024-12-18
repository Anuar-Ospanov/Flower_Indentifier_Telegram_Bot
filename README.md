# Flower Identification Bot 🌸

This project is a Telegram bot that uses AI and machine learning to identify flowers from images and provide interesting facts about them. The bot utilizes a pre-trained ONNX model for flower classification and OpenAI's GPT-4 model to generate flower-related information.

## Features

- 🌼 **Flower Identification**: Send a picture of a flower, and the bot will identify it using a pre-trained model.
- 📚 **Flower Facts**: The bot will fetch and provide interesting facts about the identified flower.
- 🌸 **User-Friendly Interaction**: Simple commands to start the bot and get help.

## Photos

![image](https://github.com/user-attachments/assets/291d45a1-6bbb-4c48-9bff-6ca73b21211f)
![image](https://github.com/user-attachments/assets/07f6075d-7fe4-4c01-9a68-943ba78b2fe3)
![image](https://github.com/user-attachments/assets/c32e803d-ce8b-470d-9d54-5b88a50f8be4)

## Requirements

- Python 3.7+
- `pip` (for installing dependencies)

## Installation

1. **Clone the repository**:

   ```bash
   git clone https://github.com/Anuar-Ospanov/flower-identification-bot.git
   cd flower-identification-bot
   ```

2. **Create a virtual environment (optional but recommended)**:

   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows use `venv\Scripts\activate`
   ```

3. **Install dependencies**:

   ```bash
   pip install -r requirements.txt
   ```

4. **Set up environment variables**:

   - Create a `.env` file in the root directory and add your OpenAI API key:
     ```plaintext
     OPENAI_API_KEY=your_openai_api_key_here
     ```
   - Alternatively, you can set the `OPENAI_API_KEY` as an environment variable:
     ```bash
     export OPENAI_API_KEY=your_openai_api_key_here
     ```

5. **Run the bot**:
   ```bash
   python bot.py
   ```

## Usage

- **Start the bot**: Send `/start` to the bot, and it will guide you through the process.
- **Send a flower image**: The bot will identify the flower and provide interesting facts.
- **Help**: Send `/help` to get more details on how to use the bot.

## API & Models

- **OpenAI GPT-4**: Used to fetch flower-related information.
- **ONNX Model**: A pre-trained model (ResNet50) for flower classification.

## Contributing

1. Fork the repository
2. Create a new branch (`git checkout -b feature-name`)
3. Commit your changes (`git commit -am 'Add new feature'`)
4. Push to the branch (`git push origin feature-name`)
5. Create a new Pull Request

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
