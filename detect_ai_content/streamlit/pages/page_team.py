import streamlit as st
import base64
import os
import pathlib

# Set page configuration as the first Streamlit command
st.set_page_config(page_title="TrueNet – Our Team", layout="wide")

# Convert the image to Base64
def get_base64_image(file_path):
    with open(file_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")

def app():
    # Prepare base64 string for the logo
    parent_path = pathlib.Path(__file__).parent.parent.resolve()
    save_path = os.path.join(parent_path, "data")
    logo_base64 = get_base64_image(f"{save_path}/logo3_transparent.png")

    # Consolidated Custom CSS
    st.markdown(
    """
    <style>

    .title-custom {
        font-size: 36px;
        font-weight: bold;
        margin-bottom: 40px;
        font-family: 'Arial', sans-serif;
        text-align: left;
    }
    .title-custom span {
        color: #65c6ba;
    }

    .team-container {
        text-align: center;
        margin-bottom: 40px; /* Add consistent spacing between members */
    }
    .team-container img {
        border-radius: 50% !important; /* Ensures round images */
        margin-bottom: 10px;
        max-width: 150px;
        height: 150px;
        object-fit: cover !important; /* Prevents distortion */
        display: block;
        box-shadow: 0px 4px 8px rgba(0, 0, 0, 0.1); /* Optional shadow */
    }
    .team-caption {
        font-size: 20px;
        font-weight: bold;
        margin-top: 10px;
    }
    .team-caption span {
        color: #65c6ba; /* Last name in green */
    }
    .team-role {
        font-size: 16px;
        line-height: 1.8;
        max-width: 220px;
        margin: 0 auto !important; /* Centers text relative to the image */
        text-align: center !important; /* Align text in the center */
        margin-top: 10px;
        margin-bottom: 25px; /* Adds space below */
    }

    hr {
        border: 1px solid #e0e0e0; /* Light gray */
        margin-top: 40px; /* Space before footer */
        margin-bottom: 10px;
    }
    footer {
        text-align: center;
        margin-top: 20px;
        padding-bottom: 20px;
    }
    </style>
    """,
    unsafe_allow_html=True
    )

   # Display custom header with logo and left-aligned title
    st.markdown(f"""
        <div style="background-color:#f5f5f5; padding:10px; border-radius:10px; margin-bottom:10px; text-align: left;">
            <img src="data:image/png;base64,{logo_base64}" alt="TrueNet Logo" style="height:60px; width:auto; margin-left: 0;">
        </div>
    """, unsafe_allow_html=True)

    # Title with colored "TrueNet" aligned to the left
    st.markdown('<div class="title-custom" style="text-align: left;">Meet the people behind <span>TrueNet</span></div>', unsafe_allow_html=True)

    # Display each team member in separate columns
    team_members = [
        {"name": "YUKA <span>KUDO</span>", "role": "Team Leader and Data Engineer at German Biology Institute. Based in Berlin.", "image": "yuka.png"},
        {"name": "JÉRÔME <span>MORISSARD</span>", "role": "iOS Fullstack Developer. Entrepreneur. Based in Biarritz.", "image": "jerome.png"},
        {"name": "ABHINAV <span>BANERJEE</span>", "role": "Tech Consultant at Deloitte. Based in London.", "image": "abhi.png"},
        {"name": "MOHAMED <span>TARAWALLI</span>", "role": "Accountant at KPMG. Based in Nigeria", "image": "achmed.png"},
        {"name": "LINA <span>PALACIOS</span>", "role": "Mathematician at Milliman. Based in Munich.", "image": "lina.png"}
    ]

    # Create columns dynamically
    columns = st.columns(len(team_members))  # Create a column for each team member

    for col, member in zip(columns, team_members):
        with col:
            st.markdown('<div class="team-container">', unsafe_allow_html=True)
            st.image(os.path.join(save_path, member["image"]), width=150, use_column_width=False)  # Simplified st.image
            st.markdown(f'<div class="team-caption">{member["name"]}</div>', unsafe_allow_html=True)
            st.markdown(f'<div class="team-role">{member["role"]}</div>', unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)

     # Add a horizontal line before the footer
    st.markdown("<hr>", unsafe_allow_html=True)

    # Footer
    st.markdown("""
        <footer>
            © 2024 TrueNet AI Detection. All rights reserved.
        </footer>
    """, unsafe_allow_html=True)

# This allows the team page to be run as a standalone app for testing
if __name__ == "__main__":
    st.sidebar.title('Navigation')
    app()
