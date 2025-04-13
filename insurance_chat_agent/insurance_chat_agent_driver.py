# Databricks notebook source
# MAGIC %md
# MAGIC # Building a multi-agent system with OpenAI Agents SDK on Databricks
# MAGIC
# MAGIC This notebook is the companion to the blog post [Building a multi-agent system with OpenAI Agents SDK on Databricks](https://medium.com/@AI-on-Databricks/building-a-multi-agent-system-with-openai-agents-sdk-on-databricks-6b6ad6774477)
# MAGIC
# MAGIC OpenAI recently released its own [Agents SDK](https://github.com/openai/openai-agents-python/tree/main) — it is a lightweight yet powerful framework for building multi-agent workflows.
# MAGIC
# MAGIC In this notebook, we demonstrate how to leverage the OpenAI Agents SDK and Databricks’ features to build and deploy an insurance-policy Q&A agent on the Databricks Data Intelligence Platform.
# MAGIC
# MAGIC
# MAGIC **Note**:
# MAGIC
# MAGIC * The notebook is best to be run in a Unity Catalog enabled Databricks Workspace with serverless compute cluster
# MAGIC * The example datasets are provided [here](https://github.com/qian-yu-db/OpenAI_Agents_SDK_on_Databricks/tree/main/datasets)

# COMMAND ----------

# MAGIC %pip install -r ./requirements.txt
# MAGIC %restart_python

# COMMAND ----------

import mlflow
import os
import warnings

# Pull your OpenAI API key from Databricks secrets
my_scrept_scope = "TO BE REPLACED"
os.environ["OPENAI_API_KEY"] = dbutils.secrets.get(scope=f"my_secret_scope", key="OpenAI")

warnings.filterwarnings("ignore", category=UserWarning)

# Set MLflow Experiment
my_databricks_account = "TO BE REPLACED"
mlflow.set_experiment(f"/Users/{my_databricks_account}/ML_experiments/insurance_chat_agent")
mlflow.openai.autolog()

# COMMAND ----------

# MAGIC %md
# MAGIC ## Set up UC Tools
# MAGIC
# MAGIC The key components of an Agentic system are LLM(s), Tools/knowledge Base(s), Orchastrator, and evaluation.
# MAGIC
# MAGIC ![image](../imgs/multi-agent-workflow.webp)
# MAGIC
# MAGIC The UC tools can be convieniently created by using SQL syntax as below. We create the following UC functions for our AI insurance agent:
# MAGIC
# MAGIC - `search_claims_details_by_policy_no`: a SQL function that retrieves a customer’s claim history based on their policy number. Since it is governed by UC, it follows the 3-level namespace when defining the name of the UC function.
# MAGIC - `policy_docs_vector_search`: a Databricks SQL AI Function that retrieves relevant document chunks from a Databricks Vector Search index using the approximate nearest neighbor (ANN) algorithm on an input query. Alternatively one can also define this tool using the Databricks AI Bridge `VectorSearchRetrieverTool`. See more details [`here`](https://docs.databricks.com/aws/en/generative-ai/agent-framework/unstructured-retrieval-tools).
# MAGIC
# MAGIC
# MAGIC Please replace `catalog`, `schema` with your UC catalog and schema names.
# MAGIC
# MAGIC ```sql
# MAGIC -- Create a SQL function that searches for claims details by policy_no
# MAGIC CREATE OR REPLACE FUNCTION catalog.schema.search_claims_details_by_policy_no (
# MAGIC     input_policy_no STRING COMMENT 'Policy number'
# MAGIC )
# MAGIC RETURNS TABLE
# MAGIC COMMENT 'Returns policy details about a customer given policy_no.'
# MAGIC RETURN
# MAGIC SELECT *
# MAGIC FROM catalog.schema.claims_table
# MAGIC WHERE policy_no = input_policy_no
# MAGIC ;
# MAGIC
# MAGIC -- Create a SQL function that calls the vector_search() AI Function
# MAGIC CREATE OR REPLACE FUNCTION catalog.schema.policy_docs_vector_search (
# MAGIC     query STRING
# MAGIC     COMMENT 'The query string for searching insurance policy documentation.'
# MAGIC )
# MAGIC RETURNS TABLE
# MAGIC COMMENT 'Executes a search on insurance policy documentation to retrieve text documents most relevant to the input query.'
# MAGIC RETURN
# MAGIC SELECT
# MAGIC     chunked_text as page_content,
# MAGIC     map('doc_path', path, 'chunk_id', chunk_id) as metadata
# MAGIC FROM
# MAGIC     vector_search(
# MAGIC         index => catalog.schema.policy_docs_chunked_files_vs_index',
# MAGIC         query => query,
# MAGIC         num_results => 3
# MAGIC     )
# MAGIC ;
# MAGIC ```
# MAGIC
# MAGIC To make UC functions work as tools with OpenAI Agents SDK, We uses [unity catalog OpenAI integration](https://github.com/unitycatalog/unitycatalog/tree/main/ai/integrations/openai) to perform UC function calls and they need to be wrapped under the `@function_tool` decorator.
# MAGIC
# MAGIC In addition, it supports Pydantic as a way to inject contexts into the agent, tool, or handoffs, or to enforce output type from an agent. Here we create a `UserInfo` class with a Pydantic base model with customer ID and policy number as arguments, which is used as context. The `conversation_id` and `user_id` are context parameters for MLflow `ChatAgent`, they are included so that we can log this workflow as a `ChatAgent` model.
# MAGIC
# MAGIC

# COMMAND ----------

from pydantic import BaseModel

class UserInfo(BaseModel):
    cust_id: str | None = None
    policy_no: str | None = None
    conversation_id: str | None = None
    user_id: str | None = None

# COMMAND ----------

# MAGIC %md
# MAGIC ### Define Tools

# COMMAND ----------

from unitycatalog.ai.core.databricks import (
    DatabricksFunctionClient,
    FunctionExecutionResult,
)
from agents import function_tool, RunContextWrapper


@function_tool
def search_claims_details_by_policy_no(wrapper: RunContextWrapper[UserInfo], policy_no: str) -> FunctionExecutionResult:
    print("[DEBUG]: the 'search_claims_details_by_policy_no' tool was called")
    wrapper.context.policy_no = policy_no
    client = DatabricksFunctionClient()
    return client.execute_function(
        function_name="ai.insurance_agent.search_claims_details_by_policy_no",
        parameters={"input_policy_no": wrapper.context.policy_no},
    )

@function_tool
def policy_docs_vector_search(query: str) -> FunctionExecutionResult:
    print("[DEBUG]: the 'policy_docs_vector_search' tool was called")
    client = DatabricksFunctionClient()
    return client.execute_function(
        function_name="ai.insurance_agent.policy_docs_vector_search",
        parameters={"query": query},
    )

# COMMAND ----------

# MAGIC %md
# MAGIC ## Create Agents
# MAGIC
# MAGIC A multi-agent system is like a conversation between LLMs that work together to solve a problem, which requires “handing off” prompts, contexts, and outputs from one LLM (i.e. a single agent) to another.
# MAGIC
# MAGIC
# MAGIC Here is a visualization of the multi-agent system
# MAGIC
# MAGIC ![image](../imgs/insurance_chat_agent.png)

# COMMAND ----------

# If you want custom model hosted outside of OpenAI
# You can use the following to set up
from openai import AsyncOpenAI
from agents import OpenAIChatCompletionsModel, set_tracing_disabled

API_KEY = (
    dbutils.notebook.entry_point.getDbutils()
    .notebook()
    .getContext()
    .apiToken()
    .getOrElse(None)
)
BASE_URL = (
    f'https://{spark.conf.get("spark.databricks.workspaceUrl")}/serving-endpoints'
)

# You can replace 'gtp-4o' with the MODEL variable in the Agent definition
MODEL = "databricks-claude-3-7-sonnet"

client = AsyncOpenAI(
    base_url=BASE_URL,
    api_key=API_KEY,
)


# COMMAND ----------

# MAGIC %md
# MAGIC The instruction field is the prompt for the agent. We’ve added a `RECOMMENDED_PROMPT_PREFIX` object from OpenAI. This is a optional “hand-off” prompt, and we found it to work fairly well. The tools field is where you designate a list of tools that the agent can use to complete the task specified in the instruction. The model field is where you specify the LLM that powers the agent. It works natively with all OpenAI models, provided that you have access to them

# COMMAND ----------

from agents.extensions.handoff_prompt import RECOMMENDED_PROMPT_PREFIX

# This is OpenAI SDK's recommended prompt prefix
RECOMMENDED_PROMPT_PREFIX

# COMMAND ----------

from agents import Agent, Runner, set_tracing_disabled

# You can turn off tracing by setting this to True
set_tracing_disabled(disabled=False)

claims_detail_retrieval_agent = Agent[UserInfo](
    name="Claims Details Retrieval Agent",
    instructions=(
        f"{RECOMMENDED_PROMPT_PREFIX}"
        "You are a claims details retrieval agent. "
        "If you are speaking to a customer, you probably were transferred to you from the triage agent. "
        "Use the following routine to support the customer. \n"
        "# Routine: \n"
        "1. Identify the last question asked by the customer. \n"
        "2. Use the search tools to retrieve data about a claim. Do not rely on your own knowledge. \n"
        "3. If you cannot answer the question, transfer back to the triage agent. \n"
    ),
    tools=[
        search_claims_details_by_policy_no,
    ],
    model="gpt-4o",
)

policy_qa_agent = Agent[UserInfo](
    name="Policy Q&A Agent",
    instructions=(
        f"{RECOMMENDED_PROMPT_PREFIX}"
        "You are an insurance policy Q&A agent. "
        "If you are speaking to a customer, you probably were transferred to you from the triage agent. "
        "Use the following routine to support the customer.\n"
        "# Routine: \n"
        "1. Identify the last question asked by the customer. \n"
        "2. Use the search tools to answer the question about their policy. Do not rely on your own knowledge. \n"
        "3. If you cannot answer the question, transfer back to the triage agent. \n"
    ),
    tools=[policy_docs_vector_search],
    model="gpt-4o",
)

triage_agent = Agent[UserInfo](
    name="Triage agent",
    instructions=(
        f"{RECOMMENDED_PROMPT_PREFIX}"
        "You are a helpful triaging agent. "
        "You can use your tools to delegate questions to other appropriate agents. "
        "If the customer does not have anymore questions, wish them a goodbye and a good rest of their day. "
    ),
    handoffs=[claims_detail_retrieval_agent, policy_qa_agent],
    model="gpt-4o",
)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Experiment with MLflow

# COMMAND ----------

# Input some user data as context
user_info = UserInfo(cust_id="1234", policy_no="12345678", coversation_id="123", user_id="123")
user_input = "[USER]: I'like to check on my existing claims"

# COMMAND ----------

# Start a chat span
with mlflow.start_span(name="insurance_agent", span_type="AGENT") as span:
    print("[AGENT] Hello! How may I assist you?")
    while True:
        user_input = input("[USER]: ")
        if user_input.lower() == "exit":
            print("[AGENT]: Bye!")
            break
        if not user_input:
            continue
        try:
            result = await Runner.run(
                starting_agent=triage_agent, input=user_input, context=user_info
            )
            print("\n[AGENT]:", result.final_output)
        except Exception as e:
            print(f"\nError occurred: {str(e)}")

# COMMAND ----------

result = await Runner.run(triage_agent, input=user_input, context=user_info)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Log, Register and Deploy with Mosaic Agent Framework
# MAGIC
# MAGIC In order to deploy the chatbot as a “model”, and serve it on Databricks Model Serving, there are a few key steps:
# MAGIC
# MAGIC - Wrap your agent code in a mlflow.pyfunc.ChatAgent interface as a custom “model”. This schema specification is designed for agent scenarios and is similar to, but not strictly compatible with, the OpenAI ChatCompletion schema.
# MAGIC - Log and register the multi-agent system to MLflow and to UC.
# MAGIC - Use the `agents.deploy()` function to deploy your chatbot as a REST API endpoint, and instantiate a Review App to begin the evaluation loop.

# COMMAND ----------

# MAGIC %md
# MAGIC ### Write the agent chat model to file to prepare for logging the model
# MAGIC
# MAGIC

# COMMAND ----------

# MAGIC %%writefile insurance_chat_agent.py
# MAGIC from typing import Any, List, Optional, Dict, Generator
# MAGIC from mlflow.pyfunc import ChatAgent
# MAGIC from mlflow.entities import SpanType
# MAGIC from mlflow.types.agent import (
# MAGIC     ChatAgentChunk,
# MAGIC     ChatAgentMessage,
# MAGIC     ChatAgentResponse,
# MAGIC     ChatContext,
# MAGIC )
# MAGIC import mlflow
# MAGIC from uuid import uuid4
# MAGIC import asyncio
# MAGIC from pydantic import BaseModel
# MAGIC from unitycatalog.ai.core.databricks import (
# MAGIC     DatabricksFunctionClient,
# MAGIC     FunctionExecutionResult,
# MAGIC )
# MAGIC from agents import function_tool, RunContextWrapper
# MAGIC from agents import Agent, Runner, set_tracing_disabled
# MAGIC from agents.extensions.handoff_prompt import RECOMMENDED_PROMPT_PREFIX
# MAGIC
# MAGIC # os.environ["OPENAI_API_KEY"] = "{{secrets/databricks_token_qyu/OpenAi}}"
# MAGIC mlflow.openai.autolog()
# MAGIC
# MAGIC class UserInfo(BaseModel):
# MAGIC     cust_id: str | None = None
# MAGIC     policy_no: str | None = None
# MAGIC     conversation_id: str | None = None
# MAGIC     user_id: str | None = None
# MAGIC
# MAGIC
# MAGIC @function_tool
# MAGIC def search_claims_details_by_policy_no(wrapper: RunContextWrapper[UserInfo], policy_no: str) -> FunctionExecutionResult:
# MAGIC     print("[DEBUG]: the 'search_claims_details_by_policy_no' tool was called")
# MAGIC     wrapper.context.policy_no = policy_no
# MAGIC     client = DatabricksFunctionClient()
# MAGIC     return client.execute_function(
# MAGIC         function_name="ai.insurance_agent.search_claims_details_by_policy_no",
# MAGIC         parameters={"input_policy_no": wrapper.context.policy_no},
# MAGIC     )
# MAGIC
# MAGIC
# MAGIC @function_tool
# MAGIC def policy_docs_vector_search(query: str) -> FunctionExecutionResult:
# MAGIC     print("[DEBUG]: the 'policy_docs_vector_search' tool was called")
# MAGIC     client = DatabricksFunctionClient()
# MAGIC     return client.execute_function(
# MAGIC         function_name="ai.insurance_agent.policy_docs_vector_search",
# MAGIC         parameters={"query": query},
# MAGIC     )
# MAGIC
# MAGIC set_tracing_disabled(disabled=False)
# MAGIC
# MAGIC claims_detail_retrieval_agent = Agent[UserInfo](
# MAGIC     name="Claims Details Retrieval Agent",
# MAGIC     instructions=(
# MAGIC         f"{RECOMMENDED_PROMPT_PREFIX}"
# MAGIC         "You are a claims details retrieval agent. "
# MAGIC         "If you are speaking to a customer, you probably were transferred to you from the triage agent. "
# MAGIC         "Use the following routine to support the customer. \n"
# MAGIC         "# Routine: \n"
# MAGIC         "1. Identify the last question asked by the customer. \n"
# MAGIC         "2. Use the search tools to retrieve data about a claim. Do not rely on your own knowledge. \n"
# MAGIC         "3. If you cannot answer the question, transfer back to the triage agent. \n"
# MAGIC     ),
# MAGIC     tools=[
# MAGIC         search_claims_details_by_policy_no,
# MAGIC     ],
# MAGIC     model="gpt-4o",
# MAGIC )
# MAGIC
# MAGIC policy_qa_agent = Agent[UserInfo](
# MAGIC     name="Policy Q&A Agent",
# MAGIC     instructions=(
# MAGIC         f"{RECOMMENDED_PROMPT_PREFIX}"
# MAGIC         "You are an insurance policy Q&A agent. "
# MAGIC         "If you are speaking to a customer, you probably were transferred to you from the triage agent. "
# MAGIC         "Use the following routine to support the customer.\n"
# MAGIC         "# Routine: \n"
# MAGIC         "1. Identify the last question asked by the customer. \n"
# MAGIC         "2. Use the search tools to answer the question about their policy. Do not rely on your own knowledge. \n"
# MAGIC         "3. If you cannot answer the question, transfer back to the triage agent. \n"
# MAGIC     ),
# MAGIC     tools=[policy_docs_vector_search],
# MAGIC     model="gpt-4o",
# MAGIC )
# MAGIC
# MAGIC triage_agent = Agent[UserInfo](
# MAGIC     name="Triage agent",
# MAGIC     instructions=(
# MAGIC         f"{RECOMMENDED_PROMPT_PREFIX}"
# MAGIC         "You are a helpful triaging agent. "
# MAGIC         "You can use your tools to delegate questions to other appropriate agents. "
# MAGIC         "If the customer does not have anymore questions, wish them a goodbye and a good rest of their day. "
# MAGIC     ),
# MAGIC     handoffs=[claims_detail_retrieval_agent, policy_qa_agent],
# MAGIC     model="gpt-4o",
# MAGIC )
# MAGIC
# MAGIC class InsuranceChatAgent(ChatAgent):
# MAGIC     def __init__(self, starting_agent: Agent):
# MAGIC         self.starting_agent = starting_agent
# MAGIC
# MAGIC     def _convert_to_input_text(selfself, messages: List[ChatAgentMessage]) -> str:
# MAGIC         """Extract the most recent user messages as input text"""
# MAGIC         for message in reversed(messages):
# MAGIC             if message.role == "user":
# MAGIC                 return message.content
# MAGIC             return ""
# MAGIC     
# MAGIC     def _create_user_context(
# MAGIC             self, 
# MAGIC             context: Optional[ChatContext] = None, 
# MAGIC             custom_inputs: Optional[Dict[str, Any]] = None
# MAGIC         ) -> UserInfo:
# MAGIC         """Convert MLflow inputs to UserInfo object"""
# MAGIC         user_info = UserInfo()
# MAGIC         
# MAGIC         if context:
# MAGIC             conversation_id = getattr(context, "conversation_id", None)
# MAGIC             if conversation_id:
# MAGIC                 user_info.conversation_id = conversation_id
# MAGIC                 
# MAGIC             user_id = getattr(context, "user_id", None)
# MAGIC             if user_id:
# MAGIC                 user_info.user_id = user_id
# MAGIC             
# MAGIC         return user_info
# MAGIC
# MAGIC     @mlflow.trace(name="insurance_chat_agent", span_type=SpanType.AGENT)
# MAGIC     def predict(
# MAGIC         self,
# MAGIC         messages: list[ChatAgentMessage],
# MAGIC         context: Optional[ChatContext] = None,
# MAGIC         custom_inputs: Optional[Dict[str, Any]] = None
# MAGIC     ) -> ChatAgentResponse:
# MAGIC         input_text = self._convert_to_input_text(messages)
# MAGIC         user_info = self._create_user_context(context, custom_inputs)
# MAGIC
# MAGIC         # Run the agent use asyncio
# MAGIC         loop = asyncio.new_event_loop()
# MAGIC         asyncio.set_event_loop(loop)
# MAGIC         try:
# MAGIC             result = loop.run_until_complete(
# MAGIC                 Runner.run(
# MAGIC                     starting_agent=self.starting_agent,
# MAGIC                     input=input_text,
# MAGIC                     context=user_info,
# MAGIC                 )
# MAGIC             )
# MAGIC         finally:
# MAGIC             loop.close()
# MAGIC
# MAGIC         # Convert the result to ChatAgentResponse format:
# MAGIC         return ChatAgentResponse(
# MAGIC             messages=[
# MAGIC                 ChatAgentMessage(
# MAGIC                     role="assistant",
# MAGIC                     content=result.final_output,
# MAGIC                     id=str(uuid4())
# MAGIC                 )
# MAGIC             ]
# MAGIC         )
# MAGIC
# MAGIC     @mlflow.trace(name="insurance_change_agent_stream", span_type=SpanType.AGENT)
# MAGIC     def predict_stream(
# MAGIC         self,
# MAGIC         messages: list[ChatAgentMessage],
# MAGIC         context: Optional[ChatContext] = None,
# MAGIC         custom_inputs: Optional[Dict[str, Any]] = None
# MAGIC     ) -> Generator[ChatAgentResponse, None, None]:
# MAGIC         response = self.predict(messages, context, custom_inputs)
# MAGIC
# MAGIC         # Yield it as a single chunk
# MAGIC         for message in response.messages:
# MAGIC             yield ChatAgentChunk(delta=message)
# MAGIC
# MAGIC AGENT = InsuranceChatAgent(starting_agent=triage_agent)
# MAGIC mlflow.models.set_model(AGENT)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Load from agent python code and run a couple of validations

# COMMAND ----------

dbutils.library.restartPython()

# COMMAND ----------

from insurance_chat_agent import AGENT
import os
import nest_asyncio

nest_asyncio.apply()
os.environ["OPENAI_API_KEY"] = dbutils.secrets.get(
    scope="my_secret_scope", key="OpenAi"
)

AGENT.predict(
    {
        "messages": [
            {
                "role": "user",
                "content": "hi, id like to check on my existing claims and my policy number: 102070455",
            }
        ],
        "context": {"conversation_id": "123", "user_id": "123"},
    }
)

# COMMAND ----------

AGENT.predict({
        "messages": [{"role": "user", "content": "does my policy cover towing and labor costs?"}],
        "context": {"conversation_id": "123", "user_id": "123"}
})

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC ### Logging the model
# MAGIC
# MAGIC To enable automatic authentication passthrough, since the chatbot needs access to data on Databricks, [specify dependent resources](https://docs.databricks.com/aws/en/generative-ai/agent-framework/log-agent#-specify-resources-for-automatic-authentication-passthrough-system-authentication) using the resources parameter of the MLflow log_model() API.

# COMMAND ----------

import mlflow
import os
from mlflow.models.resources import (
    DatabricksFunction,
    DatabricksServingEndpoint,
    DatabricksVectorSearchIndex)
from unitycatalog.ai.openai.toolkit import UCFunctionToolkit
import nest_asyncio

nest_asyncio.apply()
os.environ["OPENAI_API_KEY"] = dbutils.secrets.get(
    scope="my_secret_scope", key="OpenAi"
)

resources = [
    DatabricksVectorSearchIndex(
        index_name="ai.agents.policy_docs_chunked_files_vs_index"
    ),
    DatabricksServingEndpoint(endpoint_name="databricks-bge-large-en"),
    DatabricksFunction(
        function_name="ai.insurance_agent.search_claims_details_by_policy_no"
    ),
    DatabricksFunction(
        function_name="ai.insurance_agent.policy_docs_vector_search"
    )
]

mlflow.set_experiment(f"/Users/{my_databricks_account}/ML_experiments/insurance_chat_agent")
mlflow.openai.autolog()

# Log the model using the "models from code" approach
with mlflow.start_run():
    logged_model_info = mlflow.pyfunc.log_model(
        artifact_path="insurance_chat_agent",
        python_model=os.path.join(os.getcwd(), "insurance_chat_agent.py"),
        input_example={
            "messages": [
                {
                    "role": "user",
                    "content": "hi, id like to check on my existing claims?",
                }
            ],
            "context": {"conversation_id": "123", "user_id": "123"},
        },
        pip_requirements=[
            "mlflow",
            "openai-agents",
            "unitycatalog-openai[databricks]==0.2.0",
            "pydantic",
        ],
        resources=resources
    )

# COMMAND ----------

# MAGIC %md
# MAGIC ### Test logged the model

# COMMAND ----------

import nest_asyncio

nest_asyncio.apply()

# Load the model
loaded_model = mlflow.pyfunc.load_model(logged_model_info.model_uri)

# Test it with a sample input
response = loaded_model.predict({
        "messages": [{"role": "user", "content": "hi, id like to check on my existing claims?"}],
        "context": {"conversation_id": "123", "user_id": "123"}
})

print(response)

# COMMAND ----------

response = loaded_model.predict({
        "messages": [{"role": "user", "content": "hi, id like to check on my existing claims and my policy number is 102070455"}],
        "context": {"conversation_id": "123", "user_id": "123"}
})

# COMMAND ----------

response = loaded_model.predict({
        "messages": [{"role": "user", "content": "does my policy cover towing and labor costs?"}],
        "context": {"conversation_id": "123", "user_id": "123"}
})

# COMMAND ----------

# MAGIC %md
# MAGIC ### Registered the model

# COMMAND ----------

mlflow.set_registry_uri("databricks-uc")

catalog = "ai"
schema = "agents"
model_name = "insurance_chat_agent"
UC_MODEL_NAME = f"{catalog}.{schema}.{model_name}"

# register the model to UC
uc_registered_model_info = mlflow.register_model(
    model_uri=logged_model_info.model_uri, name=UC_MODEL_NAME
)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Deploy to an endpoint

# COMMAND ----------

# MAGIC %md
# MAGIC #### Pre-deployment Env Test

# COMMAND ----------

mlflow.models.predict(
    model_uri=f"runs:/{logged_model_info.run_id}/insurance_chat_agent",
    input_data={
        "messages": [{"role": "user", "content": "hi, id like to check on my existing claims?"}],
        "context": {"conversation_id": "123", "user_id": "123"}
        },
    env_manager="uv"
)

# COMMAND ----------

# MAGIC %md
# MAGIC #### Deploy

# COMMAND ----------

from databricks import agents

agents.deploy(
    UC_MODEL_NAME,
    uc_registered_model_info.version,
    environment_vars={
        "OPENAI_API_KEY": "{{secrets/my_secret_scope/OpenAI}}",
        "DATABRICKS_TOKEN": "{{secrets/my_secret_scope/databricks_token}}",
    },
    tags={"endpoint_desc": "insurance_chat_agent_openai_agent_sdk"},
)
