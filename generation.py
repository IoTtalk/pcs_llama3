from llm_server import get_llm, get_tokenizer, get_sampling_params


def generate_with_loop(model_path, message, histories):
    """
    Generate answer according to chat histories and newly given message.

    Parameters:
        model_path (str): The model path (both directory name of HuggingFace and local relative path are available).
        message (str): The new query input to generate answer.
        histories (list): The chat histories including contents from both human(user) and assistant(llm).

    Returns:
        str: A generated answer which is remaining fulfilled.
    """
    
    history = []
    
    # Add all chat content from both human and assistant.
    for human, assistant in histories:
        history.append({"role": "user", "content": human})
        history.append({"role": "assistant", "content": assistant})
    # Add message into the list and mark its role as user.
    history.append({"role": "user", "content": message})
    
    # =====Setting Here=====
    # Choose a version of llama3 from HuggingFace.
    llm = get_llm(model_path)
    tokenizer = get_tokenizer()
    sampling_params = get_sampling_params()
    
    prompt = tokenizer.apply_chat_template(history, tokenize=False)
    
    # Keep return the newest result generated from llm.
    for chunk in llm.generate(prompt, sampling_params):
        yield chunk.outputs[0].text


# Run this python file independently to try the function defined before.
if __name__ == "__main__":
    user_query = "What is Anthracnose caused by?"
    
    histories = []
    
    # Call function "generate_with_loop" to generate answer using llama3.
    generated_answer = generate_with_loop("meta-llama/Llama-3.2-3B-Instruct", user_query, histories)
    
    answer = ""
    
    # Keep update answer until the whole answer has been generated.
    for ans in generated_answer:
        answer = ans
    
    print(answer)
