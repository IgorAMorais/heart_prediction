import pandas as pd
import streamlit as st
from joblib import dump, load

modelo = load("model.pkl")
print('hello world')
