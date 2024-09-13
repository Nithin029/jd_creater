import os
import json
import bcrypt
from datetime import datetime
from functools import lru_cache

import streamlit as st
from dotenv import load_dotenv
import psycopg2
from openai import OpenAI
from langchain_community.vectorstores import Neo4jVector
from neo4j import GraphDatabase
from langchain_together.embeddings import TogetherEmbeddings

load_dotenv()

# Constants and configurations
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
TOGETHER_API_KEY = os.getenv("TOGETHER_API_KEY")
NEO4J_URI = os.getenv("uri")
NEO4J_USERNAME = "neo4j"
NEO4J_PASSWORD = os.getenv("password")
POSTGRES_USER = os.getenv("user")
POSTGRES_PASSWORD = os.getenv("password_postgres")

client = OpenAI(
    base_url="https://api.together.xyz/v1",
    api_key=TOGETHER_API_KEY
)
embeddings = TogetherEmbeddings(model="BAAI/bge-base-en-v1.5", together_api_key=TOGETHER_API_KEY)
SysPromptDefault = "You are now in the role of an expert AI."


# Load prompts
with open('prompt.json', 'r') as file:
    prompts = json.load(file)

# Vector indices setup
@st.cache_resource
def setup_vector_indices():
    return {
        "jobs_title_embedding": Neo4jVector.from_existing_index(
            embedding=embeddings,
            url=NEO4J_URI,
            username=NEO4J_USERNAME,
            password=NEO4J_PASSWORD,
            search_type="hybrid",
            index_name="jobs_title_embedding",
            keyword_index_name="title_index",
            embedding_node_property="title_embedding",
            text_node_property="title",
        ),
        "companies": Neo4jVector.from_existing_index(
            embedding=embeddings,
            url=NEO4J_URI,
            username=NEO4J_USERNAME,
            password=NEO4J_PASSWORD,
            search_type="hybrid",
            index_name="companies",
            keyword_index_name="company_index",
            embedding_node_property="embedding",
            text_node_property="company_name",
        ),
        "locations": Neo4jVector.from_existing_index(
            embedding=embeddings,
            url=NEO4J_URI,
            username=NEO4J_USERNAME,
            password=NEO4J_PASSWORD,
            search_type="hybrid",
            index_name="locations",
            keyword_index_name="location_index",
            embedding_node_property="embedding",
            text_node_property="location_name",
        ),
        "jobtypes": Neo4jVector.from_existing_index(
            embedding=embeddings,
            url=NEO4J_URI,
            username=NEO4J_USERNAME,
            password=NEO4J_PASSWORD,
            search_type="hybrid",
            index_name="jobtypes",
            keyword_index_name="jobtype_index",
            embedding_node_property="embedding",
            text_node_property="jobtype",
        ),
    }

vector_indices = setup_vector_indices()

# Database connections
@st.cache_resource
def get_neo4j_driver():
    return GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USERNAME, NEO4J_PASSWORD))

@st.cache_resource
def get_postgres_connection():
    return psycopg2.connect(
        dbname="postgres",
        user=POSTGRES_USER,
        password=POSTGRES_PASSWORD,
        host="aws-0-us-east-1.pooler.supabase.com",
        port="6543"
    )

# Helper functions
def set_default_value(value, default="None"):
    return default if value is None or value.strip() == "" else value

@st.cache_data
def response(message: object, model: object = "llama3-8b-8192", SysPrompt: object = SysPromptDefault,
             temperature: object = 0.2) -> object:
    """

    :rtype: object
    """
    client = OpenAI(
        api_key=GROQ_API_KEY,
        base_url="https://api.groq.com/openai/v1",
    )

    messages = [{"role": "system", "content": SysPrompt}, {"role": "user", "content": message}]
    response = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=temperature,
        frequency_penalty=0.2,
    )
    return response.choices[0].message.content

@st.cache_data
def get_job_type():
    with get_neo4j_driver().session() as session:
        query = "MATCH (n:JobType) RETURN DISTINCT n.type as type"
        result = session.run(query)
        return [record["type"] for record in result]

@st.cache_data
def get_job_titles(company_name):
    with get_neo4j_driver().session() as session:
        query = """
        MATCH (c:Company {name: $company_name})-[:POSTED]->(j:Job)
        RETURN DISTINCT j.title AS title
        """
        result = session.run(query, company_name=company_name)
        return [record["title"] for record in result]

@st.cache_data
def get_job_description(company_name, job_title):
    with get_neo4j_driver().session() as session:
        query = """
        MATCH (c:Company {name: $company_name})-[:POSTED]->(j:Job)
        WHERE j.title = $job_title OR j.cleansed_title = $job_title
        RETURN j.description AS description
        LIMIT 1
        """
        result = session.run(query, company_name=company_name, job_title=job_title)
        return result.single()["description"] if result.peek() else "No description found."

@st.cache_data
def get_locations_for_company(company_name):
    with get_neo4j_driver().session() as session:
        query = """
        MATCH (c:Company {name: $company_name})-[:LOCATION]->(l:Location)
        RETURN DISTINCT l.location AS location
        """
        result = session.run(query, company_name=company_name)
        return [record['location'] for record in result]

@st.cache_data
def get_salary_ranges_for_company(company_name):
    with get_neo4j_driver().session() as session:
        query = """
        MATCH (c:Company {name: $company_name})-[:POSTED]->(j:Job)
        WHERE j.salary_estimate IS NOT NULL
        RETURN DISTINCT j.salary_estimate AS salary_range
        """
        result = session.run(query, company_name=company_name)
        return [record['salary_range'] for record in result]

def filter_locations(locations):
    filtered_locations = []
    for location in locations:
        parts = location.split(', ')
        if len(parts) == 3:
            city = parts[0]
            state = parts[2]
            filtered_locations.append(f"{city}, {state}")
    return filtered_locations

@st.cache_data
def get_sample_job_description_from_neo4j(company_name):
    with get_neo4j_driver().session() as session:
        query = """
        MATCH (c:Company {name: $company_name})-[:POSTED]->(j:Job)
        RETURN j.description AS description
        LIMIT 1
        """
        result = session.run(query, company_name=company_name)
        record = result.single()
        return record["description"] if record else None

# Authentication functions
def register_company(company_name, password):
    try:
        hashed_password = bcrypt.hashpw(password.encode(), bcrypt.gensalt()).decode()
        with get_postgres_connection() as conn:
            with conn.cursor() as cur:
                cur.execute("""
                CREATE TABLE IF NOT EXISTS companies (
                    company_name VARCHAR(255) PRIMARY KEY,
                    password VARCHAR(255) NOT NULL
                );
                """)
                cur.execute("INSERT INTO companies (company_name, password) VALUES (%s, %s);",
                            (company_name, hashed_password))
                conn.commit()
        return True
    except psycopg2.IntegrityError:
        return False
    except Exception as e:
        st.error(f"Error registering company: {e}")
        return False

def authenticate(company_name, password):
    try:
        with get_postgres_connection() as conn:
            with conn.cursor() as cur:
                cur.execute("SELECT password FROM companies WHERE company_name = %s;", (company_name,))
                result = cur.fetchone()
                if result and bcrypt.checkpw(password.encode(), result[0].encode()):
                    return True
        return False
    except Exception as e:
        st.error(f"Error authenticating company: {e}")
        return False

# Job handling functions
def handle_existing_title(company_name, job_title):
    old_description = get_job_description(company_name, job_title)
    handle_job_listing(company_name, job_title, old_description)

def handle_new_title(company_name, new_title):
    st.subheader("New Job Title")
    st.write(f"Creating new job listing for: {new_title}")

    # Perform similarity search on the new title
    similar_titles = vector_indices["jobs_title_embedding"].similarity_search_with_score(new_title)

    # Use a dictionary to store unique titles and their first description
    unique_titles = {}
    for doc, _ in similar_titles:
        title = doc.metadata['cleansed_title']
        if title not in unique_titles:
            unique_titles[title] = doc.metadata.get('description', None)

    cleansed_titles = list(unique_titles.keys())
    cleansed_titles.append("Enter custom title")  # Add option for custom title
    selected_option = st.selectbox("Select a job title or enter a new one", cleansed_titles)

    if selected_option == "Enter custom title":
        custom_title = st.text_input("Enter your custom job title")
        if custom_title:
            handle_custom_title(company_name, custom_title)
        return

    current_company_job_description = get_sample_job_description_from_neo4j(company_name)
    alternative_description = unique_titles[selected_option] if selected_option in unique_titles else None

    if alternative_description:
        st.write("Using combined description from similar job and company sample.")
    else:
        st.write("No similar titles found, using a sample job description from the company.")

    handle_job_listing(company_name, new_title, current_company_job_description, alternative_description)

def handle_custom_title(company_name, custom_title):
    current_company_sample_description = get_sample_job_description_from_neo4j(company_name)
    handle_job_listing(company_name, custom_title, current_company_sample_description,is_custom=True)

def handle_job_listing(company_name, job_title, old_description, alternative_description=None,is_custom=False):
    locations = get_locations_for_company(company_name)
    filtered_locations = filter_locations(locations)
    salaries = get_salary_ranges_for_company(company_name)
    jobtypes = get_job_type()

    # Common input fields
    selected_location = st.selectbox("Select a location", filtered_locations + ["Enter new location"])
    if selected_location == "Enter new location":
        selected_location = st.text_input("Enter new location (county, city, state)")

    selected_salary = st.selectbox("Select a salary range", salaries + ["Enter new salary range"])
    if selected_salary == "Enter new salary range":
        selected_salary = st.text_input("Enter new salary range")

    new_requirements = st.text_area("Enter job requirements (optional)")
    selected_department = st.text_input("Enter Department (optional)")
    working_days = st.text_input("Enter Working Days (optional)")
    selected_jobtype = st.selectbox("Select a jobtype", jobtypes + ["Enter new jobtype"])
    if selected_jobtype == "Enter new jobtype":
        selected_jobtype = st.text_input("Enter new jobtype")
    additional_benefits = st.text_area("Enter Additional Benefits/Details (optional)")

    if st.button("Generate Job Description"):
        if is_custom:
            message = f"""
                    COMPANY_SAMPLE_DESCRIPTION:
                    {old_description}

                    NEW_REQUIREMENTS:
                    {new_requirements}

                    JOB_TITLE:
                    {job_title}

                    LOCATION:
                    {selected_location}

                    SALARY:
                    {selected_salary}

                    DEPARTMENT:
                    {selected_department}

                    WORKING_DAYS:
                    {working_days}

                    ADDITIONAL_BENEFITS:
                    {additional_benefits}

                    JOB_TYPE:
                    {selected_jobtype}
                    """
            sys_prompt = prompts['jd_new']
        elif alternative_description:
            message = f"""
            ALTERNATE_COMPANY_DESCRIPTION:
            {alternative_description}

            CURRENT_COMPANY_SAMPLE_DESCRIPTION:
            {old_description}

            NEW_REQUIREMENTS:
            {new_requirements}

            JOB_TITLE:
            {job_title}

            LOCATION:
            {selected_location}

            SALARY:
            {selected_salary}

            DEPARTMENT:
            {selected_department}

            WORKING_DAYS:
            {working_days}

            ADDITIONAL_BENEFITS:
            {additional_benefits}

            JOB_TYPE:
            {selected_jobtype}
            """
            sys_prompt = prompts['jd_alternative']
        else:
            message = f"""
            PREVIOUS_JD:
            {old_description}

            NEW_REQUIREMENTS:
            {new_requirements}

            JOB_TITLE:
            {job_title}

            LOCATION:
            {selected_location}

            SALARY:
            {selected_salary}

            DEPARTMENT:
            {selected_department}

            WORKING_DAYS:
            {working_days}

            ADDITIONAL_BENEFITS:
            {additional_benefits}

            JOB_TYPE:
            {selected_jobtype}
            """
            sys_prompt = prompts['jd']

        model = "llama-3.1-70b-versatile"
        st.session_state.generated_description = response(message=message, model=model, SysPrompt=sys_prompt, temperature=0.2)

    if st.session_state.get('generated_description'):
        st.subheader("Generated Job Description (Editable)")
        updated_description = st.text_area("Edit Job Description", st.session_state.generated_description, height=300)

        if st.button("Create New Job Listing"):
            requirements_to_insert = new_requirements if new_requirements.strip() else "None"
            if insert_job_data(company_name, job_title, alternative_description, old_description, updated_description,
                               requirements_to_insert, selected_location, selected_salary, selected_department,
                               working_days, additional_benefits, selected_jobtype):
                st.success(f"New job listing created for {job_title}!")
            else:
                st.error("Failed to create job listing.")

def insert_job_data(company_name, job_title, alternative_company_description, old_description, generated_description, requirements, location, salary, department, working_days, additional_benefits, job_type):
    try:
        with get_postgres_connection() as conn:
            with conn.cursor() as cur:
                cur.execute("""
                CREATE TABLE IF NOT EXISTS job_listings (
                    id SERIAL PRIMARY KEY,
                    company_name VARCHAR(255),
                    job_title VARCHAR(255),
                    alternative_company_description TEXT,
                    old_description TEXT,
                    generated_description TEXT,
                    requirements TEXT,
                    location VARCHAR(255),
                    salary VARCHAR(255),
                    department TEXT,
                    work VARCHAR(511),
                    additional TEXT,
                    jobtype VARCHAR(255),
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                );
                """)
                cur.execute("""
                INSERT INTO job_listings (company_name, job_title, alternative_company_description, old_description, generated_description, requirements, location, salary, department, work, additional, jobtype)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s);
                """, (company_name, job_title, alternative_company_description, old_description, generated_description, requirements, location, salary, department, working_days, additional_benefits, job_type))
                conn.commit()
        return True
    except Exception as e:
        st.error(f"Error inserting job data: {e}")
        return False

# Main Streamlit app
def main():
    st.title("Company Job Portal")

    if 'logged_in' not in st.session_state:
        st.session_state.logged_in = False

    if not st.session_state.logged_in:
        st.subheader("Login or Register")
        action = st.radio("Choose an action", ["Login", "Register"])

        company_name = st.text_input("Company Name")
        password = st.text_input("Password", type="password")

        if action == "Login":
            if st.button("Login"):
                if authenticate(company_name, password):
                    st.session_state.logged_in = True
                    st.session_state.company_name = company_name
                    st.rerun()
                else:
                    st.error("Invalid credentials")
        else:  # Register
            if st.button("Register"):
                if register_company(company_name, password):
                    st.success("Registration successful! You can now log in.")
                else:
                    st.error("Company already exists")
    else:
        st.write(f"Welcome, {st.session_state.company_name}!")
        if st.button("Logout"):
            st.session_state.logged_in = False
            st.experimental_rerun()

        job_titles = get_job_titles(st.session_state.company_name)
        job_titles.append("Enter new title")
        selected_option = st.selectbox("Select a job title or enter a new one", job_titles)

        if selected_option == "Enter new title":
            new_title = st.text_input("Enter new job title")
            if new_title:
                handle_new_title(st.session_state.company_name, new_title)
        elif selected_option:
            handle_existing_title(st.session_state.company_name, selected_option)


if __name__ == "__main__":
    main()