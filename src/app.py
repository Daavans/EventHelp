"""Streamlit chat interface for the Event Assistant."""

import streamlit as st

from src.agent import AgentResult, run_agent

EXAMPLE_QUESTIONS = [
    "How much does a beer cost?",
    "Is my ticket RF100003 still valid?",
    "Can I still park on Saturday?",
    "Mag mijn zoontje van 14 mee?",
    "I'd like to join the waitlist for a weekend ticket",
    "What is the capital of France?",
]

st.set_page_config(page_title="Riverside Festival Assistant", page_icon="🎵")
st.title("🎵 Riverside Festival Assistant")
st.caption("Ask me anything about Riverside Festival — Tiel, 16–18 July 2027.")

# Streamlit reruns this whole script on every interaction, so the chat lives in session_state.
if "messages" not in st.session_state:
    st.session_state.messages = []  # {"role": "user"/"assistant", "content": str, "result": AgentResult}


def show_details(result: AgentResult) -> None:
    """Show which FAQ chunks and tools the agent used (handy for demos)."""
    with st.expander("🔍 How did I get this answer?"):
        st.markdown("**FAQ chunks used** (similarity score)")
        if result.faq_chunks:
            for chunk, score in result.faq_chunks:
                st.markdown(f"- `{score:.3f}` {chunk.splitlines()[0].removeprefix('## ')}")
        else:
            st.markdown("- none above the relevance threshold")

        st.markdown("**Tool calls**")
        if result.tool_calls:
            for call in result.tool_calls:
                st.markdown(f"- `{call['name']}`")
                st.json({"args": call["args"], "result": call["result"]}, expanded=False)
        else:
            st.markdown("- none")


with st.sidebar:
    st.header("Try an example")
    clicked_example = None
    for example in EXAMPLE_QUESTIONS:
        if st.button(example, use_container_width=True):
            clicked_example = example
    st.divider()
    if st.button("🗑️ Clear chat", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

# Show the conversation so far.
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if message.get("result"):
            show_details(message["result"])

question = st.chat_input("Ask a question about the festival...") or clicked_example

if question:
    # Only plain text goes into the history; FAQ context is retrieved fresh for each question.
    history = [{"role": m["role"], "content": m["content"]} for m in st.session_state.messages]

    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            try:
                result = run_agent(question, history)
            except Exception as e:  # never show a stack trace to a visitor
                print(f"Agent error: {e}")
                result = AgentResult(
                    answer="Sorry, something went wrong on our side. Please try again in a moment."
                )
        st.markdown(result.answer)
        show_details(result)

    st.session_state.messages.append({"role": "assistant", "content": result.answer, "result": result})
