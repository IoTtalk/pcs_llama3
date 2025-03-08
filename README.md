# Llama 3 on Your Local Computer

Run the Llama 3 model (8-B or 70B) on your server / computer.

## Getting Started

### Installation

1. Clone the repository:
```
git clone https://github.com/IoTtalk/pcs_llama3.git
cd llama3_local
```

2. Install required dependencies:
```
pip install -r requirements.txt
```

notify that python3.12 cannot be used.

## Usage

If you want to run the server and make sure that it can work on your mechine, you can just run the command below.

```
python llm_server.py
```

This will open a chat bot server, and you can open the website in any browser and chat with the LLM model.  As for setting up in a linux server for long term using, you are suggested to use tmux to run code.

If you want to call the LLM model within API, you can run the command below.

```
python generation.py
```

In this python file, all the functions in llm_server.py which are related to model setting are included and packed into a API function.  The main function of generation.py demonstrates how to call the API function.  Place generation.py into your project which needs to use LLM and modify the relative path of function importing at the first line, then you can use the API with customized LLM at anywhere.
