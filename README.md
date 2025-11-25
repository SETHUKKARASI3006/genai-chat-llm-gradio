## Development and Deployment of a 'Chat with LLM' Application Using the Gradio Blocks Framework

### AIM:
To design and deploy a "Chat with LLM" application by leveraging the Gradio Blocks UI framework to create an interactive interface for seamless user interaction with a large language model.

### PROBLEM STATEMENT:

The challenge is to move beyond a simple single-turn question/answer interface and design a **multi-turn conversational** UI. This requires: 1) capturing and formatting **chat history** into a context-aware prompt for the LLM; 2) implementing the UI using the flexible **Gradio `Blocks`** framework with a dedicated **`gr.Chatbot`** component; and 3) adding optional **advanced controls** (like system messages and temperature) for a richer user experience.

-----

### DESIGN STEPS:

#### STEP 1:
Install libraries (like `text_generation` and `gradio`), configure the **Hugging Face API key**, and initialize the **LLM client** (e.g., FalconLM) for communication.

#### STEP 2:
Create an initial **`gr.Interface`** demo with a **`gr.Textbox`** prompt and a **`gr.Slider`** for `max_new_tokens` to verify the core LLM connection.

#### STEP 3:
Switch to the **`gr.Blocks`** framework and implement the conversational UI using the **`gr.Chatbot`** component, text input, and submit/clear buttons.

#### STEP 4:
Develop the **`format_chat_prompt`** function to serialize the user's new message and the **`chat_history`** list into the specific multi-turn format required by the LLM.

#### STEP 5:
Update the **`respond`** function to use the contextual prompt, call the LLM, append the LLM's response to the **`chat_history`**, and then update the **`gr.Chatbot`**.

#### STEP 6:
Integrate **System Instruction** (**`gr.Textbox`**) and **Temperature** (**`gr.Slider`**) controls, group them using **`gr.Accordion`**, and implement **response streaming** for real-time output.

### PROGRAM:

```
import os
import io
import IPython.display
from PIL import Image
import base64 
import requests 
requests.adapters.DEFAULT_TIMEOUT = 60

from dotenv import load_dotenv, find_dotenv
_ = load_dotenv(find_dotenv()) # read local .env file
hf_api_key = os.environ['HF_API_KEY']
# Helper function
import requests, json
from text_generation import Client

#FalcomLM-instruct endpoint on the text_generation library
client = Client(os.environ['HF_API_FALCOM_BASE'], headers={"Authorization": f"Basic {hf_api_key}"}, timeout=120)
prompt = "Has math been invented or discovered?"
client.generate(prompt, max_new_tokens=256).generated_text
import gradio as gr
def generate(input, slider):
    output = client.generate(input, max_new_tokens=slider).generated_text
    return output

demo = gr.Interface(fn=generate, 
                    inputs=[gr.Textbox(label="Prompt"), 
                            gr.Slider(label="Max new tokens", 
                                      value=20,  
                                      maximum=1024, 
                                      minimum=1)], 
                    outputs=[gr.Textbox(label="Completion")])

gr.close_all()
demo.launch(share=True, server_port=int(os.environ['PORT1']))
import random

def respond(message, chat_history):
        #No LLM here, just respond with a random pre-made message
        bot_message = random.choice(["Tell me more about it", 
                                     "Cool, but I'm not interested", 
                                     "Hmmmm, ok then"]) 
        chat_history.append((message, bot_message))
        return "", chat_history

with gr.Blocks() as demo:
    chatbot = gr.Chatbot(height=240) #just to fit the notebook
    msg = gr.Textbox(label="Prompt")
    btn = gr.Button("Submit")
    clear = gr.ClearButton(components=[msg, chatbot], value="Clear console")

    btn.click(respond, inputs=[msg, chatbot], outputs=[msg, chatbot])
    msg.submit(respond, inputs=[msg, chatbot], outputs=[msg, chatbot]) #Press enter to submit

gr.close_all()
demo.launch(share=True, server_port=int(os.environ['PORT2']))
def format_chat_prompt(message, chat_history):
    prompt = ""
    for turn in chat_history:
        user_message, bot_message = turn
        prompt = f"{prompt}\nUser: {user_message}\nAssistant: {bot_message}"
    prompt = f"{prompt}\nUser: {message}\nAssistant:"
    return prompt

def respond(message, chat_history):
        formatted_prompt = format_chat_prompt(message, chat_history)
        bot_message = client.generate(formatted_prompt,
                                     max_new_tokens=1024,
                                     stop_sequences=["\nUser:", "<|endoftext|>"]).generated_text
        chat_history.append((message, bot_message))
        return "", chat_history

with gr.Blocks() as demo:
    chatbot = gr.Chatbot(height=240) #just to fit the notebook
    msg = gr.Textbox(label="Prompt")
    btn = gr.Button("Submit")
    clear = gr.ClearButton(components=[msg, chatbot], value="Clear console")

    btn.click(respond, inputs=[msg, chatbot], outputs=[msg, chatbot])
    msg.submit(respond, inputs=[msg, chatbot], outputs=[msg, chatbot]) #Press enter to submit

gr.close_all()
demo.launch(share=True, server_port=int(os.environ['PORT3']))
def format_chat_prompt(message, chat_history, instruction):
    prompt = f"System:{instruction}"
    for turn in chat_history:
        user_message, bot_message = turn
        prompt = f"{prompt}\nUser: {user_message}\nAssistant: {bot_message}"
    prompt = f"{prompt}\nUser: {message}\nAssistant:"
    return prompt
def respond(message, chat_history, instruction, temperature=0.7):
    prompt = format_chat_prompt(message, chat_history, instruction)
    chat_history = chat_history + [[message, ""]]
    stream = client.generate_stream(prompt,
                                      max_new_tokens=1024,
                                      stop_sequences=["\nUser:", "<|endoftext|>"],
                                      temperature=temperature)
                                      #stop_sequences to not generate the user answer
    acc_text = ""
    #Streaming the tokens
    for idx, response in enumerate(stream):
            text_token = response.token.text

            if response.details:
                return

            if idx == 0 and text_token.startswith(" "):
                text_token = text_token[1:]

            acc_text += text_token
            last_turn = list(chat_history.pop(-1))
            last_turn[-1] += acc_text
            chat_history = chat_history + [last_turn]
            yield "", chat_history
            acc_text = ""
with gr.Blocks() as demo:
    chatbot = gr.Chatbot(height=240) #just to fit the notebook
    msg = gr.Textbox(label="Prompt")
    with gr.Accordion(label="Advanced options",open=False):
        system = gr.Textbox(label="System message", lines=2, value="A conversation between a user and an LLM-based AI assistant. The assistant gives helpful and honest answers.")
        temperature = gr.Slider(label="temperature", minimum=0.1, maximum=1, value=0.7, step=0.1)
    btn = gr.Button("Submit")
    clear = gr.ClearButton(components=[msg, chatbot], value="Clear console")

    btn.click(respond, inputs=[msg, chatbot, system], outputs=[msg, chatbot])
    msg.submit(respond, inputs=[msg, chatbot, system], outputs=[msg, chatbot]) #Press enter to submit

gr.close_all()
demo.queue().launch(share=True, server_port=int(os.environ['PORT4']))
gr.close_all()
```

### OUTPUT:

<img width="594" height="594" alt="image" src="https://github.com/user-attachments/assets/21c483ed-cd9a-4135-86e4-331fddd29305" />

<img width="575" height="590" alt="image" src="https://github.com/user-attachments/assets/ba839636-1ca8-4b07-b6a4-03b8fb92115b" />

<img width="594" height="577" alt="image" src="https://github.com/user-attachments/assets/0834b18c-5cf8-4c75-a1c8-bf29df67e5ef" />

<img width="581" height="569" alt="image" src="https://github.com/user-attachments/assets/09d37084-844e-4edb-aa84-1c89d0c26c18" />

### RESULT:
Thus, a a "Chat with LLM" application by leveraging the Gradio Blocks UI framework is designed and deployed successfully.
