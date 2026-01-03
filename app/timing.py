import time
import streamlit as st

def reset_timings():
    st.session_state.timings = {}

class UITimer:
    def __init__(self, name: str):
        self.name = name

    def __enter__(self):
        self.start = time.perf_counter()

    def __exit__(self, exc_type, exc, tb):
        elapsed = (time.perf_counter() - self.start) * 1000
        st.session_state.timings[self.name] = elapsed