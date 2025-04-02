from typing import Any, List, Optional, Dict, Generator
from mlflow.pyfunc import ChatAgent
from mlflow.entities import SpanType
from mlflow.types.agent import (
    ChatAgentChunk,
    ChatAgentMessage,
    ChatAgentResponse,
    ChatContext,
)
import mlflow
from uuid import uuid4
import asyncio
from pydantic import BaseModel
from unitycatalog.ai.core.databricks import (
    DatabricksFunctionClient,
    FunctionExecutionResult,
)
from agents import function_tool, RunContextWrapper
from agents import Agent, Runner, set_tracing_disabled
from agents.extensions.handoff_prompt import RECOMMENDED_PROMPT_PREFIX

mlflow.openai.autolog()

class UserInfo(BaseModel):
    cust_id: str | None = None
    policy_no: str | None = None
    conversation_id: str | None = None
    user_id: str | None = None


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

class InsuranceChatAgent(ChatAgent):
    def __init__(self, starting_agent: Agent):
        self.starting_agent = starting_agent

    def _convert_to_input_text(selfself, messages: List[ChatAgentMessage]) -> str:
        """Extract the most recent user messages as input text"""
        for message in reversed(messages):
            if message.role == "user":
                return message.content
            return ""
    
    def _create_user_context(
            self, 
            context: Optional[ChatContext] = None, 
            custom_inputs: Optional[Dict[str, Any]] = None
        ) -> UserInfo:
        """Convert MLflow inputs to UserInfo object"""
        user_info = UserInfo()
        
        if context:
            conversation_id = getattr(context, "conversation_id", None)
            if conversation_id:
                user_info.conversation_id = conversation_id
                
            user_id = getattr(context, "user_id", None)
            if user_id:
                user_info.user_id = user_id
            
        return user_info

    @mlflow.trace(name="insurance_chat_agent", span_type=SpanType.AGENT)
    def predict(
        self,
        messages: list[ChatAgentMessage],
        context: Optional[ChatContext] = None,
        custom_inputs: Optional[Dict[str, Any]] = None
    ) -> ChatAgentResponse:
        input_text = self._convert_to_input_text(messages)
        user_info = self._create_user_context(context, custom_inputs)

        # Run the agent use asyncio
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            result = loop.run_until_complete(
                Runner.run(
                    starting_agent=self.starting_agent,
                    input=input_text,
                    context=user_info,
                )
            )
        finally:
            loop.close()

        # Convert the result to ChatAgentResponse format:
        return ChatAgentResponse(
            messages=[
                ChatAgentMessage(
                    role="assistant",
                    content=result.final_output,
                    id=str(uuid4())
                )
            ]
        )

    @mlflow.trace(name="insurance_change_agent_stream", span_type=SpanType.AGENT)
    def predict_stream(
        self,
        messages: list[ChatAgentMessage],
        context: Optional[ChatContext] = None,
        custom_inputs: Optional[Dict[str, Any]] = None
    ) -> Generator[ChatAgentResponse, None, None]:
        response = self.predict(messages, context, custom_inputs)

        # Yield it as a single chunk
        for message in response.messages:
            yield ChatAgentChunk(delta=message)

AGENT = InsuranceChatAgent(starting_agent=triage_agent)
mlflow.models.set_model(AGENT)
