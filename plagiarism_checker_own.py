import streamlit as st
import yaml
import os
import pandas as pd
import requests
import base64
import ast
import time
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
# ---------------------------
# Load credentials from YAML
# ---------------------------
def load_credentials(file_path='users.yml'):
    with open(file_path, 'r') as file:
        data = yaml.safe_load(file)
        return {user['username']: user for user in data['users']}

USER_DB = load_credentials()

# ---------------------------
# Session State for Navigation
# ---------------------------
if "page" not in st.session_state:
    st.session_state.page = "login"

# Logout button
#if st.session_state.page == "main":
#    if st.button("Logout"):
#        st.session_state.page = "login"
#        st.session_state["authenticated"] = False
#        st.rerun()

# ---------------------------
# Login Page
# ---------------------------
if st.session_state.page == "login":
    st.title("Sign In")
    username = st.text_input("Username")
    password = st.text_input("Password", type="password")

    if st.button("Login"):
        if username in USER_DB and USER_DB[username]["password"] == password:
            st.session_state["authenticated"] = True
            st.session_state["role"] = USER_DB[username]["role"]
            st.session_state.page = "main"
            st.rerun()
        else:
            st.error("Invalid username or password")
# ---------------------------
# Guardrail: Validate Python Code
# ---------------------------
def is_valid_python_code(content):
    try:
        ast.parse(content)
        return True
    except SyntaxError:
        return False

# ---------------------------
# Similarity Check Function
# ---------------------------
def check_similarity(target_text, source_text):
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform([target_text, source_text])
    return cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0]

# ---------------------------
# Keyword Extraction for GitHub Search
# ---------------------------
def extract_keywords(code_text, max_keywords=5):
    words = [w for w in code_text.split() if len(w) > 3]
    return " ".join(words[:max_keywords]) if words else "python"

# ---------------------------
# Validate GitHub Token
# ---------------------------
def validate_github_token():
    if "general" not in st.secrets or "GITHUB_TOKEN" not in st.secrets["general"]:
        return False, "No GitHub token found in secrets.toml."
    
    token = st.secrets["general"]["GITHUB_TOKEN"].strip()
    headers = {"Authorization": f"token {token}"}
    r = requests.get("https://api.github.com/user", headers=headers)
    if r.status_code == 200:
        return True, "Token is valid."
    else:
        return False, f"Invalid token: {r.json().get('message', 'Unknown error')}"

# ---------------------------
# GitHub Code Fetch Function (Cached)
# ---------------------------
GITHUB_API_URL = "https://api.github.com/search/code"

@st.cache_data(show_spinner=False)
def fetch_github_code(query, max_files=5):
    token = st.secrets["general"]["GITHUB_TOKEN"].strip()
    headers = {"Authorization": f"token {token}"}
    params = {
        "q": f"{query} in:file language:python",
        "per_page": max_files
    }

    response = requests.get(GITHUB_API_URL, headers=headers, params=params)
    if response.status_code != 200:
        return []

    items = response.json().get("items", [])
    if not items:
        return []

    code_snippets = []
    for item in items:
        file_url = item["url"]
        file_data = requests.get(file_url, headers=headers).json()
        if "content" in file_data:
            content = base64.b64decode(file_data["content"]).decode("utf-8")
            code_snippets.append({"name": item["name"], "content": content})
    return code_snippets

# ---------------------------
# Generate Plagiarism Report
# ---------------------------
def generate_plagiarism_report(uploaded_file_name, target_text, threshold=80, github_results=None):
    source_folder = "source_files"
    results = []

    # Progress bar
    progress = st.progress(0)
    total_files = len(os.listdir(source_folder)) if os.path.exists(source_folder) else 0
    total_files += len(github_results) if github_results else 0
    checked = 0

    # Local files check
    if os.path.exists(source_folder):
        for source_name in os.listdir(source_folder):
            fpath = os.path.join(source_folder, source_name)
            if os.path.isfile(fpath) and source_name.endswith((".py", ".txt")):
                with open(fpath, "r", encoding="utf-8") as f:
                    source_content = f.read()
                score = check_similarity(target_text, source_content)
                percent = round(score * 100, 2)
                status = "⚠️ Flagged" if percent >= threshold else "✅ Clean"
                results.append({
                    "Uploaded File Name": uploaded_file_name,
                    "Source File Name": source_name,
                    "Similarity (%)": percent,
                    "Status": status
                })
                checked += 1
                #progress.progress(int((checked / total_files) * 100))

    # GitHub check
    if github_results:
        for item in github_results:
            score = check_similarity(target_text, item["content"])
            percent = round(score * 100, 2)
            status = "⚠️ Flagged" if percent >= threshold else "✅ Clean"
            results.append({
                "Uploaded File Name": uploaded_file_name,
                "Source File Name": f"GitHub: {item['name']}",
                "Similarity (%)": percent,
                "Status": status,
                #"Source Link": f"<a hrefiew on GitHub</a>"
            })
            checked += 1
            #progress.progress(int((checked / total_files) * 100))

    return pd.DataFrame(results)

# ---------------------------
# Main Interface
# ---------------------------
if st.session_state.page == "main" and st.session_state.get("authenticated"):
    role = st.session_state["role"]
    #st.title(f"Welcome {role}")
    with st.sidebar:
        st.markdown(f"### 👋 Welcome {role}")
        st.markdown("---")
        st.write("""
        This module helps detect plagiarism in code submissions by comparing uploaded files 
        against GitHub repositories and other sources. It calculates similarity percentages 
        and flags suspicious matches for review.
        """)
        st.markdown("---")
        if st.button("🚪 Logout"):
            st.session_state.page = "login"
            st.session_state["authenticated"] = False
            st.rerun()

    # ---- Header ----
    st.markdown(
        """
        <div style='background-color:#34495e;padding:15px;border-radius:5px;'>
            <h2 style='color:white;text-align:center;'>📚 Plagiarism Detection Module</h2>
        </div>
        """,
        unsafe_allow_html=True
    )


    # Admin uploads source files
    if role == "Admin":
        st.subheader("Upload Source Code Files")
        source_folder = "source_files"
        os.makedirs(source_folder, exist_ok=True)
        source_files = st.file_uploader("Upload Source Files", type=["py", "txt"], accept_multiple_files=True)
        if source_files:
            st.success(f"{len(source_files)} source files uploaded successfully.")
            for file in source_files:
                content = file.read().decode("utf-8")
                file_path = os.path.join(source_folder, file.name)
                mode = "a" if os.path.exists(file_path) else "w"
                with open(file_path, mode, encoding="utf-8") as f:
                    f.write(content)
    # Both roles can submit code
    if role in ["Admin", "Jury"]:
        st.subheader("Submit Code to Check")
        code_files = st.file_uploader("Upload Code File", type=["py", "txt"], accept_multiple_files=True)
        pasted_code = st.text_area("Or Paste Code Here")

        target_text = ""
        uploaded_name = ""
        all_results = []

        if code_files or pasted_code.strip():
            # Validate token once
            token_valid, token_message = validate_github_token()
            github_search = st.checkbox("Check against GitHub public repositories", disabled=not token_valid)
            #st.info(token_message)           
            if st.button("🔍 Check Plagiarism"):
                st.info("Checking plagiarism...")
                progress = st.progress(0)
                total_files = len(code_files) if code_files else 1
                checked = 0

                # Process uploaded files
                if code_files:
                    for code_file in code_files:
                        uploaded_name = code_file.name
                        target_text = code_file.read().decode("utf-8")
                        if uploaded_name.endswith(".py") and not is_valid_python_code(target_text):
                            st.error(f"❌ {uploaded_name} is not valid Python code.")
                            checked += 1
                            #progress.progress(int((checked / total_files) * 100))
                            continue
                        github_results = []
                        if github_search and token_valid:
                            query = extract_keywords(target_text)
                            github_results = fetch_github_code(query)

                        df = generate_plagiarism_report(uploaded_name, target_text, github_results=github_results)
                        all_results.append(df)

                        checked += 1
                        progress.progress(int((checked / total_files) * 100))

                # Process pasted code
                if pasted_code.strip():
                    uploaded_name = "Pasted Code"
                    target_text = pasted_code

                    if not is_valid_python_code(target_text):
                        st.error("❌ Pasted code does not look like valid Python.")
                    else:
                        github_results = []
                        if github_search and token_valid:
                            query = extract_keywords(target_text)
                            github_results = fetch_github_code(query)

                        df = generate_plagiarism_report(uploaded_name, target_text, github_results=github_results)
                        all_results.append(df)

                # Combine all reports
                if all_results:
                    final_report = pd.concat(all_results, ignore_index=True)
                    st.subheader("Plagiarism Report")
                    #st.dataframe(final_report)
                    file_status = {}
                    for file_name, group in final_report.groupby("Uploaded File Name"):
                        statuses = group["Status"].unique()
                        if len(statuses) == 1:
                            file_status[file_name] = "Clean" if "Clean" in statuses[0] else "Flagged"
                        else:
                            file_status[file_name] = "Partial"

                    source_count = len(os.listdir("source_files")) if os.path.exists("source_files") else 0
                    uploaded_count = len(file_status)
                    flagged_count = sum(1 for s in file_status.values() if s == "Flagged")
                    clean_count = sum(1 for s in file_status.values() if s == "Clean")
                    partial_count = sum(1 for s in file_status.values() if s == "Partial")

                    # ✅ Colorful Tiles with Icons
                    #st.markdown("### Summary")
                    tile_html = f"""
                    <div style='display:flex;gap:10px;flex-wrap:wrap;'>
                        <div style='flex:1;background-color:#3498db;padding:10px;border-radius:6px;color:white;text-align:center;'>
                            <h4 style='margin:5px;'>📂 Source Files</h4>
                            <p style='font-size:18px;font-weight:bold;margin:0;'>{source_count}</p>
                        </div>
                        <div style='flex:1;background-color:#9b59b6;padding:10px;border-radius:6px;color:white;text-align:center;'>
                            <h4 style='margin:5px;'>📤 Uploaded Files</h4>
                            <p style='font-size:18px;font-weight:bold;margin:0;'>{uploaded_count}</p>
                        </div>
                        <div style='flex:1;background-color:#e74c3c;padding:10px;border-radius:6px;color:white;text-align:center;'>
                            <h4 style='margin:5px;'>⚠️ Flagged</h4>
                            <p style='font-size:18px;font-weight:bold;margin:0;'>{flagged_count}</p>
                        </div>
                        <div style='flex:1;background-color:#2ecc71;padding:10px;border-radius:6px;color:white;text-align:center;'>
                            <h4 style='margin:5px;'>✅ Clean</h4>
                            <p style='font-size:18px;font-weight:bold;margin:0;'>{clean_count}</p>
                        </div>
                        <div style='flex:1;background-color:#f39c12;padding:10px;border-radius:6px;color:white;text-align:center;'>
                            <h4 style='margin:5px;'>🌓 Partially Flagged/Clean</h4>
                            <p style='font-size:18px;font-weight:bold;margin:0;'>{partial_count}</p>
                        </div>
                    </div>
                    """
                    st.markdown(tile_html, unsafe_allow_html=True)
                    st.dataframe(final_report)
                    source_folder = "source_files"
                    filtered_files = []
                    for file_name, group in final_report.groupby("Uploaded File Name"):
                        max_similarity = group["Similarity (%)"].max()
                        status = file_status[file_name]

                        if status == "Clean" and max_similarity < 50:
                            filtered_files.append(file_name)

                    st.write("### Files eligible for saving:")
                    for idx, file_name in enumerate(filtered_files):
                        # Show file name and button
                        st.write(f"**{file_name}**")
                        if st.button(f"Save {file_name}", key=f"save_{idx}"):
                            # Get content
                            content_to_save = None
                            if file_name == "Pasted Code":
                                content_to_save = pasted_code
                            else:
                                for f in code_files:
                                    if f.name == file_name:
                                        content_to_save = f.getvalue().decode("utf-8")
                                        break

                            if not content_to_save:
                                st.error(f"❌ No content found for {file_name}")
                                continue

                            # Prepare file path
                            file_path = os.path.join(source_folder, file_name)

                            # If file exists, append timestamp
                            if os.path.exists(file_path):
                                base_name, ext = os.path.splitext(file_name)
                                timestamp = time.strftime("%Y%m%d_%H%M%S")
                                new_file_name = f"{base_name}_{timestamp}{ext}"
                                file_path = os.path.join(source_folder, new_file_name)

                            # Save file
                            try:
                                with open(file_path, "w", encoding="utf-8") as f:
                                    f.write(content_to_save)
                                st.success(f"✅ Saved as {os.path.basename(file_path)}")
                                st.write(f"📂 Location: {file_path}")
                            except Exception as e:
                                st.error(f"❌ Failed to save {file_name}: {e}")
                # ✅ Add CSV Download
                    
                    csv = final_report.to_csv(index=False).encode('utf-8')
                    st.download_button(
                        label="⬇️ Download Report as CSV",
                        data=csv,
                        file_name="plagiarism_report.csv",
                        mime="text/csv"
                    )
