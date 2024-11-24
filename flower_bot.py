import os
import logging
from dotenv import load_dotenv
from telegram import Update
from telegram.ext import ApplicationBuilder, CommandHandler, MessageHandler, filters, ContextTypes
from PIL import Image
import numpy as np
import onnxruntime as ort
import torchvision.transforms as transforms
import asyncio
import nest_asyncio
import openai


logging.basicConfig(format="%(asctime)s - %(name)s - %(levelname)s - %(message)s", level=logging.INFO)

load_dotenv()



telegram_api_token = os.getenv("TELEGRAM_API_TOKEN")
openai_api_key = os.getenv("OPENAI_API_KEY")
openai.api_key = openai_api_key


transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

def preprocess_image(image_path):
    image = Image.open(image_path).convert("RGB")
    image_tensor = transform(image)
    image_tensor = image_tensor.unsqueeze(0)
    return image_tensor

def load_model(onnx_model_path):
    return ort.InferenceSession(onnx_model_path)

def predict(model, image_tensor):
    input_name = model.get_inputs()[0].name
    inputs = {input_name: image_tensor.numpy()}
    outputs = model.run(None, inputs)
    return outputs

def get_flower_info(flower_name):
    prompt = f"Tell me detailed information about the flower '{flower_name}', including interesting facts. Main point, don't mention its origin. Also your answer should be max 5-6 sentences so provide main info"
    try:
        client = openai.OpenAI(api_key=openai_api_key)
        completion = client.chat.completions.create(
            model="gpt-4o-mini",  
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": prompt}
            ]
        )
        return completion.choices[0].message.content
    except Exception as e:
        logging.error(f"Error while fetching flower information: {e}")
        return "An error occurred while trying to fetch details about the flower. Please try again later."

class_labels = [
    "🌺 Annual Mallow", "🌿 Asian Virgin's Bower", "🌷 Barbados Lily", "🌵 Bull Thistle", "🌼 Buttercup",
    "🌻 California Poppy", "🌸 Calla Lily", "🌹 Canna Lily", "🍂 Coltsfoot", "💐 Common Columbine",
    "🌺 Common Cornflag", "🌼 Common Daisy", "🌼 Common Dandelion", "🌸 Common Primroses", "🌹 Corn Poppy",
    "🌵 Desert Rose", "💠 Fritillaries", "🌺 Garden Petunia", "🌺 Passionflower", "🌸 Peruvian Lily",
    "🌸 Scarlet Beebalm", "🌻 Sunflower", "🌹 Tea Roses", "🐯 Tiger Lily", "🌺 Violets", "🌸 Wallflowers",
    "💧 Water Lilies"
]

model_path = "model/resnet50_3e4_10_secondTry2__epoch_20_accuracy_test_97.4265.onnx"
model = load_model(model_path)

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(
        "🌸 Welcome to the Flower Identification Bot! 🌸\n\n"
        "Send me a picture of a flower, and I'll tell you its name along with some interesting facts! 🌺📸"
        "Processing and Information from AI can take up to a minute. So please be patient!"
    )

async def help_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(
        "🤔 Need Help?\n\n"
        "Here’s what I can do:\n"
        "1️⃣ Send me a clear image of a flower. 🌼\n"
        "2️⃣ I’ll analyze it, tell you the flower's name, and share interesting facts about it! 🪷🌻\n\n"
        "✨ Commands you can use:\n"
        "/start - Start interacting with me 🌟\n"
        "/help - Show this help message 🛟"
    )

async def handle_image(update: Update, context: ContextTypes.DEFAULT_TYPE):
    photo = update.message.photo[-1]
    photo_file = await photo.get_file()
    image_path = f"downloads/{photo.file_id}.jpg"
    os.makedirs("downloads", exist_ok=True)
    await photo_file.download_to_drive(image_path)  

    try:
        image_tensor = preprocess_image(image_path)
        outputs = predict(model, image_tensor)
        logits = outputs[0]
        predicted_class = np.argmax(logits)

        if predicted_class < len(class_labels):
            flower_name = class_labels[predicted_class]
            flower_info = get_flower_info(flower_name)

            response = (
                f"🎉 Flower Identified! 🌸\n\n"
                f"The flower in the image is: {flower_name} 🌺✨\n\n"
                f"📖 **About {flower_name}:**\n{flower_info}\n\n"
                "🌿 Thank you for using the Flower Bot! Send another image to continue. 📸"
            )
        else:
            response = (
                "😞 Oops!\n\n"
                "I couldn’t recognize the flower. 🌾\n"
                "Please try sending a clearer image or another flower. 🌼"
            )
    except Exception as e:
        logging.error(f"Error processing image: {e}")
        response = (
            "⚠️ **Error**\n\n"
            "Something went wrong while analyzing the image. 😢\n"
            "Please try again later or contact support. 🛠️"
        )

    await update.message.reply_text(response, parse_mode="Markdown")
    os.remove(image_path)

async def main():
    logging.info("🌺 Bot is running...")
    app = ApplicationBuilder().token(telegram_api_token).build()

    app.add_handler(CommandHandler("start", start))
    app.add_handler(CommandHandler("help", help_command))
    app.add_handler(MessageHandler(filters.PHOTO, handle_image))

    await app.run_polling()

if __name__ == "__main__":
    nest_asyncio.apply() 
    asyncio.run(main())
