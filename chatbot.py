import streamlit as st
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
def load_model():
    model_name = "facebook/blenderbot-400M-distill"
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    return model, tokenizer
def generate_response(input_text, conversation_history, model, tokenizer):
    history_string = "\n".join(conversation_history)
    inputs = tokenizer.encode_plus(history_string, input_text, return_tensors="pt")
    reply_ids = model.generate(inputs["input_ids"], max_length=100, pad_token_id=tokenizer.eos_token_id)
    response = tokenizer.decode(reply_ids[0], skip_special_tokens=True)
    return response
model, tokenizer = load_model()
st.title("Interactive Chatbot")
if "conversation_history" not in st.session_state:
    st.session_state.conversation_history = []
user_input = st.text_input("Ask me anything:", "")
if user_input:
    with st.spinner("Generating response..."):#Shows a loading spinner in the UI while the chatbot generates a response.
        response = generate_response(user_input, st.session_state.conversation_history, model, tokenizer)
    st.session_state.conversation_history.append(f"You: {user_input}")
    st.session_state.conversation_history.append(f"Bot: {response}")
st.write("\n".join(st.session_state.conversation_history))



#firstofall we are creating a chatbot using streamlit UI and  hugging face packages. first we are creating the model for that we are importing the model name from huggging face
# Loads the pre-trained model from the Hugging Face model hub, and adding the tokenizerassociated with model
#after that we we are adding the generateoutput function inwhich we are adding the message history and reply message is been printed and it is decoded into the human readable format
#after this we are creating the UI
#Checks if there’s no existing conversation_history in the session state.If no conversation history exists, it initializes an empty list to store the conversation.
