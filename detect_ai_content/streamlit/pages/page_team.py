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

    # Custom CSS for styling
    st.markdown("""
        <style>
        .team-container {
            text-align: center;
            margin-bottom: 40px; /* Add space between members */
        }
        .team-image {
            margin: 0 auto;
            border-radius: 50%; /* Round borders */
        }
        .team-caption {
            font-size: 24px; /* Increased font size for names */
            font-weight: bold;
            margin-top: 10px;
            font-family: 'Arial', sans-serif;
        }
        .team-caption span {
            color: #65c6ba; /* Color for last names */
        }
        .team-role {
            font-size: 18px; /* Increased font size for roles */
            font-weight: normal;
            color: #555;
            margin-top: 10px; /* Space between name and role */
            max-width: 200px; /* Limit the width of the role description */
            margin-left: auto; /* Center text horizontally */
            margin-right: auto; /* Center text horizontally */
            line-height: 1.5; /* Improve readability */
            text-align: center;
        }
        .title-custom {
            font-size: 36px; /* Bigger title */
            font-weight: bold;
            margin-bottom: 40px; /* Add space below title */
            font-family: 'Arial', sans-serif;
            text-align: left; /* Align to the left */
        }
        .title-custom span {
            color: #65c6ba; /* Color for TrueNet in title */
        }
        footer {
            margin-top: 80px; /* Push footer lower */
        }
        </style>
    """, unsafe_allow_html=True)

    # Display custom header with logo
    st.markdown(f"""
        <div style="background-color:#f5f5f5; padding:10px; border-radius:10px; margin-bottom:10px;">
            <img src="data:image/png;base64,{logo_base64}" alt="TrueNet Logo" style="height:60px; width:auto;">
        </div>
    """, unsafe_allow_html=True)

    # Title with colored "TrueNet"
    st.markdown('<div class="title-custom">Meet the people behind <span>TrueNet</span>:</div>', unsafe_allow_html=True)

    # Display each team member in separate columns
    team_members = [
        {"name": "YUKA <span>KUDO</span>", "role": "Team Leader and Data Engineer at German Biology Institute. Based in Berlin.", "image": "yuka.png"},
        {"name": "JÉRÔME <span>MORISSARD</span>", "role": "iOS Fullstack Developer. Entrepreneur. Based in Biarritz.", "image": "jerome.png"},
        {"name": "ABHINAV <span>BANERJEE</span>", "role": "Tech Consultant at Deloitte. Based in London.", "image": "abhi.png"},
        {"name": "MOHAMED <span>TARAWALLI</span>", "role": "Accountant at KPMG.", "image": "achmed.png"},
        {"name": "LINA <span>PALACIOS</span>", "role": "Mathematician at Milliman.", "image": "lina.png"}
    ]

    columns = st.columns(len(team_members))  # Create a column for each team member

    for col, member in zip(columns, team_members):
        with col:
            st.markdown('<div class="team-container">', unsafe_allow_html=True)
            st.image(os.path.join(save_path, member["image"]), width=150, use_column_width=False, caption="", output_format="PNG", class_="team-image")
            st.markdown(f'<div class="team-caption">{member["name"]}</div>', unsafe_allow_html=True)
            st.markdown(f'<div class="team-role">{member["role"]}</div>', unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)

    # Footer
    st.markdown("---")
    st.markdown("""
        <footer style="text-align:center;">
            © 2024 TrueNet AI Detection. All rights reserved.
        </footer>
    """, unsafe_allow_html=True)

# This allows the team page to be run as a standalone app for testing
if __name__ == "__main__":
    st.sidebar.title('Navigation')
    app()
