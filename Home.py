import streamlit as st 

from pathlib import Path

import streamlit as st
from st_pages import Page, add_page_title, show_pages

show_pages(
    [
        Page("Home.py", "Home", "🏠"),
        # Can use :<icon-name>: or the actual icon
        Page("pages/Project1.py", "Image Classification", "🔎"),
        Page("pages/Project2.py", "FIFA", "⚽"),
        Page("pages/Project3.py", "Time Series", "🔎"),
        Page("pages/About.py", "About", ":star:")
    ]
)

#add_page_title()  # Optional method to add title and icon to current page



html_temp = """
            <div style="background-color:#2E1A47;padding:10px">
                <nav style="display: flex; justify-content: center;">
                </nav>
            <h2 style="color:white;text-align:center;">Welcome</h2>
            </div>
            <body>
            </body>
            """
st.markdown(html_temp,unsafe_allow_html=True)