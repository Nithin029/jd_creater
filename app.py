from openai import OpenAI
from dotenv import load_dotenv
import os
import json
import psycopg2
import mysql.connector
import streamlit as st
from langchain_community.vectorstores import Neo4jVector
from neo4j import GraphDatabase
from langchain_together.embeddings import TogetherEmbeddings
import bcrypt
from datetime import datetime
load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
TOGETHER_API_KEY = os.getenv("TOGETHER_API_KEY")
uri = os.getenv("uri")
username = "neo4j"
password = os.getenv("password")
user_name = os.getenv("user_name")
password_sql=os.getenv("password_sql")
host = os.getenv("host_name")
client = OpenAI(
    base_url="https://api.together.xyz/v1",
    api_key=TOGETHER_API_KEY
)
embeddings = TogetherEmbeddings(model="BAAI/bge-base-en-v1.5", together_api_key=TOGETHER_API_KEY)
SysPromptDefault = "You are now in the role of an expert AI."

with open('prompt.json', 'r') as file:
    prompts = json.load(file)

jd = prompts['jd']
jd_alternative = prompts['jd_alternative']


vector_indices = {
    "jobs_title_embedding": Neo4jVector.from_existing_index(
        embedding=embeddings,
        url=uri,
        username=username,
        password=password,
        search_type="hybrid",
        index_name="jobs_title_embedding",
        keyword_index_name="title_index",
        embedding_node_property="title_embedding",
        text_node_property="title",
    ),
    "companies": Neo4jVector.from_existing_index(
        embedding=embeddings,
        url=uri,
        username=username,
        password=password,
        search_type="hybrid",
        index_name="companies",
        keyword_index_name="company_index",
        embedding_node_property="embedding",
        text_node_property="company_name",
    ),
    "locations": Neo4jVector.from_existing_index(
        embedding=embeddings,
        url=uri,
        username=username,
        password=password,
        search_type="hybrid",
        index_name="locations",
        keyword_index_name="location_index",
        embedding_node_property="embedding",
        text_node_property="location_name",
    ),
    "jobtypes": Neo4jVector.from_existing_index(
        embedding=embeddings,
        url=uri,
        username=username,
        password=password,
        search_type="hybrid",
        index_name="jobtypes",
        keyword_index_name="jobtype_index",
        embedding_node_property="embedding",
        text_node_property="jobtype",
    ),
}

def set_default_value(value, default="None"):
    return default if value is None or value.strip() == "" else value
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



# Neo4j connection function (for data extraction only)
def get_neo4j_driver():
    return GraphDatabase.driver(uri, auth=(username, password))

def get_job_type():
    with get_neo4j_driver().session() as session:
        query ="""
        MATCH (n:JobType) 
        RETURN DISTINCT n.type as type
        """
        result = session.run(query)
        return [record["type"] for record in result]
def get_alternate_company_with_job_title(job_title):
    with get_neo4j_driver().session() as session:
        query = """
        MATCH (c:Company)-[:POSTED]->(j:Job)
        WHERE j.title = $job_title OR j.cleansed_title = $job_title
        WITH c, j
        RETURN c.name AS company_name
        LIMIT 1
        """
        result = session.run(query, job_title=job_title)
        record = result.single()
        return record['company_name'] if record else None


def get_combined_description(company_name, job_title):
    alternate_company = get_alternate_company_with_job_title(job_title)
    print(alternate_company)
    alternate_description = None
    current_company_sample_description = None

    if alternate_company:
        alternate_description = get_job_description(alternate_company, job_title)

    current_company_sample_description = get_sample_job_description_from_neo4j(company_name)

    if not alternate_description and not current_company_sample_description:
        st.warning("No job descriptions found for the given title.")

    return alternate_description, current_company_sample_description



# PostgreSQL connection function

def get_mysql_connection():
    return mysql.connector.connect(
        database="medichire_dev",
        user=user_name,
        password=password_sql,
        host=host,
    )

    # Authentication function
def register_company(company_name, password):
    try:
        # Hash the password
        hashed_password = bcrypt.hashpw(password.encode(), bcrypt.gensalt()).decode()

        # Connect to MySQL
        conn = get_mysql_connection()
        cur = conn.cursor()

        # Insert the new company and password
        insert_query = """
        INSERT INTO companies (company_name, password) 
        VALUES (%s, %s);
        """
        cur.execute(insert_query, (company_name, hashed_password))
        conn.commit()

        cur.close()
        conn.close()

        return True
    except mysql.connector.IntegrityError:
        conn.rollback()  # In case of a duplicate entry
        return False
    except Exception as e:
        st.error(f"Error registering company: {e}")
        return False

def authenticate(company_name, password):
    try:
        # Connect to MySQL
        conn = get_mysql_connection()
        cur = conn.cursor()

        # Fetch the password from the database for the given company
        fetch_query = """
        SELECT password FROM companies 
        WHERE company_name = %s;
        """
        cur.execute(fetch_query, (company_name,))
        result = cur.fetchone()

        cur.close()
        conn.close()

        if result:
            stored_password = result[0]
            # Compare the provided password with the stored hashed password
            if bcrypt.checkpw(password.encode(), stored_password.encode()):
                return True
        return False
    except Exception as e:
        st.error(f"Error authenticating company: {e}")
        return False



# Register new company
# Function to check if the company exists in Neo4j
def company_exists_in_neo4j(company_name):
    with get_neo4j_driver().session() as session:
        query = """
        MATCH (c:Company {name: $company_name})
        RETURN c.name AS name
        LIMIT 1
        """
        result = session.run(query, company_name=company_name)
        return result.single() is not None



# Fetch job titles for a company (from Neo4j)
def get_job_titles(company_name):
    with get_neo4j_driver().session() as session:
        query = """
        MATCH (c:Company {name: $company_name})-[:POSTED]->(j:Job)
        RETURN DISTINCT j.title AS title
        """
        result = session.run(query, company_name=company_name)
        return [record["title"] for record in result]


# Fetch job description for a specific title (from Neo4j)
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


def get_sample_job_description_from_neo4j(company_name):
    with get_neo4j_driver().session() as session:
        query="""
        MATCH (c:Company {name: $company_name})-[:POSTED]->(j:Job)
        RETURN j.description AS description
        LIMIT 1
        """
        result= session.run(query,company_name=company_name)
        record = result.single()
        if record:
            return record["description"]
        return None

def get_locations_for_company(company_name):
    with get_neo4j_driver().session() as session:
        query = """
        MATCH (c:Company {name: $company_name})-[:LOCATION]->(l:Location)
        RETURN DISTINCT l.location AS location
        """
        result = session.run(query, company_name=company_name)
        return [record['location'] for record in result]


def get_salary_ranges_for_company(company_name):
    with get_neo4j_driver().session() as session:
        query = """
        MATCH (c:Company {name: $company_name})-[:POSTED]->(j:Job)
        WHERE j.salary_estimate IS NOT NULL
        RETURN DISTINCT j.salary_estimate AS salary_range
        """
        result = session.run(query, company_name=company_name)
        return [record['salary_range'] for record in result]


def insert_job_data(company_name, job_title, description, requirements, location, salary, department, working_days, additional_benefits, job_type):
    conn = None
    cur = None
    try:
        # Connect to MySQL
        conn = get_mysql_connection()
        cur = conn.cursor()

        # Handle default values
        requirements = set_default_value(requirements)
        department = set_default_value(department)
        working_days = set_default_value(working_days)
        additional_benefits = set_default_value(additional_benefits)

        # Insert job data
        insert_query = """
        INSERT INTO job_listings (company_name, job_title, description, requirements, location, salary, department, work, additional, jobtype)
        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s);
        """
        cur.execute(insert_query, (company_name, job_title, description, requirements, location, salary, department, working_days, additional_benefits, job_type))
        conn.commit()

        return True
    except (Exception, mysql.connector.Error) as error:
        print("Error while connecting to MySQL", error)
        return False
    finally:
        if cur:
            cur.close()
        if conn:
            conn.close()


# Updated function to handle new job titles
# Updated function to handle existing job titles
def handle_existing_title(company_name, job_title):
    old_description = get_job_description(company_name, job_title)

    locations = get_locations_for_company(company_name)
    salaries = get_salary_ranges_for_company(company_name)
    jobtypes=get_job_type()

    selected_location = st.selectbox("Select a location", locations + ["Enter new location"])
    if selected_location == "Enter new location":
        selected_location = st.text_input("Enter new location (county, city, state)")

    selected_salary = st.selectbox("Select a salary range", salaries + ["Enter new salary range"])
    if selected_salary == "Enter new salary range":
        selected_salary = st.text_input("Enter new salary range")

    selected_department = st.text_input("Enter Department (optional)")
    working_days = st.text_input("Enter Working Days (optional)")
    additional_benefits = st.text_area("Enter Additional Benefits/Details (optional)")
    selected_jobtype = st.selectbox("Select a jobtype", jobtypes + ["Enter new jobtype"])
    if selected_jobtype == "Enter new jobtype(optional)":
        selected_jobtype = st.text_input("Enter new jobtype (jobtype)")
    new_requirements = st.text_area("Enter updated job requirements (optional)")

    if st.button("Generate Job Description"):
        message = f"""PREVIOUS_JD:

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
        model = "llama-3.1-70b-versatile"
        st.session_state.generated_description = response(message=message, model=model, SysPrompt=jd, temperature=0.2)

    if st.session_state.get('generated_description'):
        st.subheader("Generated Job Description (Editable)")
        updated_description = st.text_area("Edit Job Description", st.session_state.generated_description, height=300)

        if st.button("Update Job Listing"):
            requirements_to_insert = new_requirements if new_requirements.strip() else "None"
            if insert_job_data(company_name, job_title, updated_description, requirements_to_insert,
                               selected_location, selected_salary,selected_department,working_days,additional_benefits,selected_jobtype):
                st.success(
                    f"Job listing updated successfully! Requirements: {'Provided' if new_requirements.strip() else 'None'}")
            else:
                st.error("Failed to update job listing.")


# Updated function to handle new job titles
def handle_new_title(company_name, new_title):
    description = None
    alternative_description = None
    current_company_job_description = None
    st.subheader("New Job Title")
    st.write(f"Creating new job listing for: {new_title}")

    # Perform similarity search on the new title
    similar_titles = vector_indices["jobs_title_embedding"].similarity_search_with_score(new_title)
    jobtypes = get_job_type()

    # Use a dictionary to store unique titles and their first description
    unique_titles = {}
    for doc, _ in similar_titles:
        title = doc.metadata['cleansed_title']
        if title not in unique_titles:
            unique_titles[title] = doc.metadata.get('description', None)

    cleansed_titles = list(unique_titles.keys())
    selected_option = st.selectbox("Select a job title or enter a new one", cleansed_titles)

    if selected_option:
        existing_titles = get_job_titles(company_name)
        if selected_option in existing_titles:
            description = get_job_description(company_name, selected_option)
        else:
            current_company_job_description = get_sample_job_description_from_neo4j(company_name)
            alternative_description = unique_titles[selected_option]
            if alternative_description:
                st.write("No exact match found for the title. Using combined description.")
                print(alternative_description)
                #print(current_company_job_description)
            else:
                st.write("No exact match found, and no alternative description available.")
    else:
        description = get_sample_job_description_from_neo4j(company_name)
        st.write("No similar titles found, using a sample job description.")

    # Rest of the function remains the same
    locations = get_locations_for_company(company_name)
    salaries = get_salary_ranges_for_company(company_name)

    selected_location = st.selectbox("Select a location", locations + ["Enter new location"])
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
        selected_jobtype = st.text_input("Enter new jobtype (jobtype)")
    additional_benefits = st.text_area("Enter Additional Benefits/Details (optional)")

    if st.button("Generate Job Description"):
        if alternative_description and current_company_job_description is not None:
            print("yes")
            message = f"""ALTERNATE_COMPANY_DESCRIPTION:

            {alternative_description}

            CURRENT_COMPANY_SAMPLE_DESCRIPTION:
            {current_company_job_description}

            NEW_REQUIREMENTS:

            {new_requirements}

            JOB_TITLE:

            {new_title}

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
            model = "llama-3.1-70b-versatile"
            st.session_state.generated_description = response(message=message, model=model, SysPrompt=jd_alternative,
                                                              temperature=0.2)
        else:
            print("no")
            message = f"""PREVIOUS_JD:

                {description}

                NEW_REQUIREMENTS:

                {new_requirements}

                JOB_TITLE:

                {new_title}

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

                I"""
            model = "llama-3.1-70b-versatile"
            st.session_state.generated_description = response(message=message, model=model, SysPrompt=jd, temperature=0.2)
    if st.session_state.get('generated_description'):
        st.subheader("Generated Job Description")
        updated_description = st.text_area("Edit Job Description", st.session_state.generated_description, height=300)

        if st.button("Create New Job Listing"):
            requirements_to_insert = new_requirements if new_requirements.strip() else "None"
            if insert_job_data(company_name, new_title, updated_description, requirements_to_insert,
                               selected_location, selected_salary,selected_department,working_days,additional_benefits,selected_jobtype):
                st.success(
                    f"New job listing created for {new_title}! Requirements: {'Provided' if new_requirements.strip() else 'None'}")
            else:
                st.error("Failed to create new job listing.")

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
