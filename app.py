import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, AIMessage
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
import streamlit as st

load_dotenv()
groq_api_key = os.getenv("CHATGROQ_API_KEY")


model = ChatGroq(model="Gemma2-9b-It", groq_api_key=groq_api_key)

store = {}

def get_session_history(session_id: str) -> BaseChatMessageHistory:
    """Retrieve or initialize session history."""
    if session_id not in store:
        store[session_id] = ChatMessageHistory()
    return store[session_id]

# Initialize message history
with_message_history = RunnableWithMessageHistory(model, get_session_history)

st.title("Chatbot with Conversation History")
st.sidebar.header("Session Options")
session_id = st.sidebar.text_input("Session ID", value="default_session")

st.header("Chat with the Bot")
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display existing messages
for message in st.session_state.messages:
    if isinstance(message, HumanMessage):
        st.write(f"**You:** {message.content}")
    elif isinstance(message, AIMessage):
        st.write(f"**Bot:** {message.content}")


user_input = st.text_input("Your message:", key="user_input")
if st.button("Send"):
    if user_input:
        # Add user's message to the conversation
        human_message = HumanMessage(content=user_input)
        st.session_state.messages.append(human_message)
        
        # Get response from the chatbot
        response = with_message_history.invoke(
            st.session_state.messages,
            config={"configurable": {"session_id": session_id}},
        )
        
        # Add AI's response to the conversation
        ai_message = AIMessage(content=response.content)
        st.session_state.messages.append(ai_message)
        
        # Display AI's response
        st.write(f"**Bot:** {response.content}")
        st.rerun()  # Rerun the app to refresh the message display

# Reset session button
if st.sidebar.button("Reset Session"):
    store[session_id] = ChatMessageHistory()
    st.session_state.messages = []
    st.success("Session reset!")