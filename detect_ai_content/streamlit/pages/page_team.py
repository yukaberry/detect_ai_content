import streamlit as st
import base64
import os
import pathlib

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
    st.markdown("""
        <style>
        /* Title */
        .title-custom {
            font-size: 36px;
            font-weight: bold;
            margin-bottom: 20px; /* Reduced spacing here */
            font-family: 'Arial', sans-serif;
            text-align: left;
        }
        .title-custom span {
            color: #65c6ba; /* Base color */
        }

        /* Team Container */
        .team-container {
            text-align: center;
            margin: 0 auto;
            padding: 20px 10px; /* Adds padding around each column */
        }

        /* Uniform image size */
        .team-container img {
            border-radius: 50%;
            width: 150px;
            height: 150px;
            object-fit: cover;
            margin-bottom: 10px;
            display: block;
            margin-left: auto; /* Center horizontally */
            margin-right: auto; /* Center horizontally */
        }

        /* Caption and role alignment */
        .team-caption {
            font-size: 18px;
            font-weight: bold;
            margin-top: 10px;
            margin-bottom: 5px;
        }

        .team-caption span {
            color: #65c6ba; /* Last names in base color */
        }

        .team-role {
            font-size: 14px;
            line-height: 1.5;
            color: #333;
            max-width: 200px;
            margin: 0 auto;
        }

        /* Footer */
        footer {
            text-align: center; /* Center the footer text */
            margin-top: 20px;
            padding-bottom: 20px;
            font-size: 14px; /* Slightly smaller font for better readability */
            color: #777777; /* Subtle gray color */
        }
        </style>
        """, unsafe_allow_html=True)


    # Display custom header with logo
    st.markdown(f"""
        <div style="background-color:#f5f5f5;
        padding:10px;
        border-radius:10px;
        margin-bottom:10px;">
            <img src="data:image/png;base64,{logo_base64}" alt="TrueNet Logo" style="height:60px; width:auto;">
        </div>
    """, unsafe_allow_html=True)

    # Title with colored "TrueNet"
    st.markdown('<div class="title-custom">Meet the people behind <span>TrueNet</span></div>', unsafe_allow_html=True)

    # Display each team member in separate columns
    # Display each team member in separate columns
    team_members = [
        {"name": "YUKA <span>KUDO</span>", "role": "Team Leader and Data Engineer at German Biology Institute. Based in Berlin.", "image": "yuka.png"},
        {"name": "JÉRÔME <span>MORISSARD</span>", "role": "iOS Fullstack Developer. Entrepreneur. Based in Biarritz.", "image": "jerome.png"},
        {"name": "ABHINAV <span>BANERJEE</span>", "role": "Tech Consultant at Deloitte. Based in London.", "image": "abhi.png"},
        {"name": "MOHAMED <span>TARAWALLI</span>", "role": "Accountant at KPMG. Based in Nigeria.", "image": "achmed.png"},
        {"name": "LINA <span>PALACIOS</span>", "role": "Mathematician at Milliman. Based in Munich.", "image": "lina.png"}
    ]

    # Create columns dynamically
    columns = st.columns(len(team_members))  # Create a column for each team member

    # Updated Code to Improve Alignment
    for col, member in zip(columns, team_members):
        with col:
            st.markdown('<div class="team-container" style="text-align:center;">', unsafe_allow_html=True)  # Center content
            st.image(
                os.path.join(save_path, member["image"]),
                width=150
            )  # Fixed size for uniformity
            st.markdown(
                f"""
                <div class="team-caption" style="text-align:center; font-weight:bold;">
                    {member["name"]}
                </div>
                <div class="team-role" style="text-align:center; font-size:14px; line-height:1.5;">
                    {member["role"]}
                </div>
                """,
                unsafe_allow_html=True
            )
            st.markdown('</div>', unsafe_allow_html=True)


    # Footer
    st.markdown("""
        <footer>
            © 2024 TrueNet AI Detection. All rights reserved.
        </footer>
    """, unsafe_allow_html=True)

app()
