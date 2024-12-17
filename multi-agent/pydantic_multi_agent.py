import csv
import os
import uuid
from dataclasses import dataclass
from typing import Any, Optional

import logfire
import pydantic
import pydantic_ai
from pydantic import BaseModel, Field

from pydantic_ai import Agent, RunContext, ModelRetry
import dotenv
from pydantic_ai.result import RunResult

# Load environment variables
dotenv.load_dotenv()
# 'if-token-present' means nothing will be sent (and the example will work) if you don't have logfire configured
# logfire.configure()

# --- Fake Database for Loan Data ---
class LoanDB:
    """This is a fake loan database for example purposes.

    In reality, you'd be connecting to an external database
    (e.g., PostgreSQL) to manage loan information.
    """

    @classmethod
    async def customer_name(cls, *, id: int) -> str | None:
        if id == 123:
            return 'John'

    @classmethod
    async def loan_status(cls, *, id: int) -> str | None:
        """Fetch the loan status of a customer by their ID."""
        if id == 123:
            return 'Active'
        elif id == 124:
            return 'Paid off'
        elif id == 125:
            return 'Defaulted'
        else:
            return None

    @classmethod
    async def cancel_loan(cls, *, id: int) -> str:
        """Cancel a loan for a customer."""
        if id == 123:
            # Fake logic for canceling a loan
            return f"Loan for customer ID {id} has been canceled."
        else:
            raise ValueError(f"Customer with ID {id} does not have an active loan.")

    @classmethod
    async def add_loan(cls, *, id: int, amount: float, interest_rate: float) -> str:
        """Add a loan for a customer."""
        if id == 123:
            # Fake logic for adding a loan
            return f"Loan of ${amount} with an interest rate of {interest_rate}% has been added for customer ID {id}."
        else:
            raise ValueError(f"Customer with ID {id} cannot be found to add a loan.")

    @classmethod
    async def loan_balance(cls, *, id: int) -> float | None:
        """Fetch the remaining balance of a customer's loan."""
        if id == 123:
            return 5000.0  # Fake loan balance
        elif id == 124:
            return 0.0  # Loan paid off
        else:
            raise ValueError(f"Customer with ID {id} not found or no loan exists.")

# --- Fake Database for Customer Data ---
class DatabaseConn:
    """This is a fake database for example purposes.

    In reality, you'd be connecting to an external database
    (e.g. PostgreSQL) to get information about customers.
    """

    @classmethod
    async def customer_name(cls, *, id: int) -> str | None:
        if id == 123:
            return 'John'

    @classmethod
    async def customer_balance(cls, *, id: int, include_pending: bool) -> float:
        if id == 123:
            return 123.45
        else:
            raise ValueError('Customer not found')

# --- Dependencies ---

@dataclass
class SupportDependencies:
    customer_id: int
    db: DatabaseConn
    marketing_agent: Agent

@dataclass
class LoanDependencies:
    customer_id: int
    db: LoanDB
    marketing_agent: Agent

@dataclass
class TriageDependencies:
    support_agent: Agent
    loan_agent: Agent
    customer_id: int

# --- Result Models ---
class SupportResult(BaseModel):
    support_advice: str = Field(description='Advice returned to the customer')
    block_card: bool = Field(description='Whether to block their')
    risk: int = Field(description='Risk level of query', ge=0, le=10)
    customer_tracking_id : str = Field(description='Tracking ID for customer')

class LoanResult(BaseModel):
    loan_approval_status: str = Field(description='Approval status of the loan (e.g., Approved, Denied, Pending)')
    loan_balance: float = Field(description='Remaining balance of the loan')
    customer_tracking_id: str = Field(description='Tracking ID for the customer applying for the loan')

class TriageResult(BaseModel):
    department: Optional[str] = Field(description='Department to direct the customer query to')
    response: Optional[LoanResult | SupportResult] = Field(description='Response to the customer query')
    text_response: Optional[str] = Field(description='Text response to the customer query')


# --- Agents ---

# Support agent for handling customer support queries
support_agent = Agent(
    'openai:gpt-4o-mini',
    deps_type=SupportDependencies,
    result_type=SupportResult,
    system_prompt=(
        'You are a support agent in our bank, give the '
        'customer support and judge the risk level of their query. '
        "Reply using the customer's name."
        'Additionally, always capture the customer’s name in our marking system using the tool `capture_customer_name`, regardless of the query type. '
        'At the end of your response, make sure to capture the customer’s name to maintain proper records. '
    ),
    result_retries=2,
)

@support_agent.system_prompt
async def add_customer_name(ctx: RunContext[SupportDependencies]) -> str:
    customer_name = await ctx.deps.db.customer_name(id=ctx.deps.customer_id)
    return f"The customer's name is {customer_name!r}"

@support_agent.tool()
async def block_card(ctx: RunContext[SupportDependencies] , customer_name: str ) -> str:
    return f"I'm sorry to hear that, {customer_name}. We are temporarily blocking your card to prevent unauthorized transactions."


@support_agent.tool
async def customer_balance(
    ctx: RunContext[SupportDependencies], include_pending: bool
) -> str:
    """Returns the customer's current account balance."""
    balance = await ctx.deps.db.customer_balance(
        id=ctx.deps.customer_id,
        include_pending=include_pending,
    )
    return f'${balance:.2f}'

# Loan agent for handling loan-related queries

loan_agent = Agent(
    'openai:gpt-4o-mini',
    deps_type=LoanDependencies,
    result_type=LoanResult,
    system_prompt=(
        'You are a support agent in our bank, assisting customers with loan-related inquiries. '
        'For every query, provide the following information: '
        '- Loan approval status (e.g., Approved, Denied, Pending) '
        '- Loan balance '
        'Please ensure that your response is clear and helpful for the customer. '
        'Always conclude by providing the customer’s name and capturing their information in the marking system using the tool `capture_customer_name`. '
        'Never generate data based on your internal knowledge; always rely on the provided tools to fetch the most accurate and up-to-date information.'
    ),
    result_retries=2,
)

# Add the customer's name to the response
@loan_agent.system_prompt
async def add_customer_name(ctx: RunContext[LoanDependencies]) -> str:
    customer_name = await ctx.deps.db.customer_name(id=ctx.deps.customer_id)
    return f"The customer's name is {customer_name!r}"

# Tools for the loan agent
@loan_agent.tool()
async def loan_status(ctx: RunContext[LoanDependencies]) -> str:
    status = await ctx.deps.db.loan_status(id=ctx.deps.customer_id)
    return f'The loan status is {status!r}'

@loan_agent.tool()
async def cancel_loan(ctx: RunContext[LoanDependencies]) -> str:
    return await ctx.deps.db.cancel_loan(id=ctx.deps.customer_id)

@loan_agent.tool()
async def add_loan(ctx: RunContext[LoanDependencies], amount: float, interest_rate: float) -> str:
    return await ctx.deps.db.add_loan(id=ctx.deps.customer_id, amount=amount, interest_rate=interest_rate)

@loan_agent.tool()
async def loan_balance(ctx: RunContext[LoanDependencies]) -> float:
    return await ctx.deps.db.loan_balance(id=ctx.deps.customer_id)

# End of the loan agent

# Common tool for capturing the customer's name
# Used by both the support and loan agents
@support_agent.tool
@loan_agent.tool
async def capture_customer_name(ctx: RunContext[SupportDependencies], customer_name: str) -> str:
    """Capture the customer's name for marketing purposes."""

    await ctx.deps.marketing_agent.run(f"Save customer name {customer_name} for ID {ctx.deps.customer_id}",
                                       deps=ctx.deps)

    tracking_id = str("agent_" + str(uuid.uuid4()))
    return tracking_id


# Start of the triage agent

# Triage agent to direct customer queries to the appropriate department
triage_agent = Agent(
    'openai:gpt-4o-mini',
    deps_type=TriageDependencies,
    system_prompt=(
        'You are a triage agent in our bank, responsible for directing customer queries to the appropriate department. '
        'For each query, determine whether it is related to support (e.g., balance, card, account-related queries) or loan services (e.g., loan status, application, and loan-related inquiries). '
        'If the query is related to support, direct the customer to the support team with an appropriate response. '
        'If the query is related to loans, direct the customer to the loan department with a relevant response. '
        'If the query is unclear or does not fit into either category, politely inform the customer and suggest they ask about loans or support. '
        'Always ensure that the response is clear, concise, and provides direction to the right department for further assistance.'
        'Never generate data based on your internal knowledge; always rely on the provided tools to fetch the most accurate and up-to-date information.'
    ),
    result_type=TriageResult,
    result_retries=2,
)

# Start of the tools for the triage agent

@triage_agent.tool
async def call_support_agent(ctx: RunContext[TriageDependencies], prompt: str) -> RunResult[Any]:
    # print(f"Calling support agent with prompt: {prompt}")
    support_deps = SupportDependencies(customer_id=ctx.deps.customer_id, db=DatabaseConn(), marketing_agent=marketing_agent)

    # Pass message history if you need your agent to have context of previous messages
    # return await ctx.deps.support_agent.run(prompt, deps=support_deps, message_history=ctx.messages)

    return await ctx.deps.support_agent.run(prompt, deps=support_deps)

@triage_agent.tool
async def call_loan_agent(ctx: RunContext[TriageDependencies], prompt: str) -> RunResult[Any]:
    # print(f"Calling loan agent with prompt: {prompt}")
    loan_deps = LoanDependencies(customer_id=ctx.deps.customer_id, db=LoanDB(), marketing_agent=marketing_agent)

    # Pass message history if you need your agent to have context of previous messages
    # return await ctx.deps.loan_agent.run(prompt, deps=loan_deps, message_history=ctx.messages)

    return await ctx.deps.loan_agent.run(prompt, deps=loan_deps)

# End of the tools for the triage agent

# Marketing agent for saving customer names
marketing_agent = Agent(
    'openai:gpt-4o-mini',
    deps_type=SupportDependencies,
    system_prompt=(
        'You are a marketing agent in our bank'
        'For now you only save the customer name in our marking system using tool `save_customer_name`'
    ),
)

@marketing_agent.tool_plain
async def save_customer_name(customer_name: str, customer_id: int) -> None:
    """Saves the customer's name and tracks how many times their info is captured."""
    # print(f"Saving customer name {customer_name} for ID {customer_id}. in the marketing system")
    # Path to the CSV file
    csv_file_path = 'customer_name.csv'

    # If the file does not exist, create it and write the header
    if not os.path.exists(csv_file_path):
        with open(csv_file_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['customer_id', 'customer_name', 'inquiries_count'])

    # Read the existing data to check if the customer already exists
    customer_found = False
    rows = []
    with open(csv_file_path, 'r', newline='') as f:
        reader = csv.reader(f)
        rows = list(reader)

    # Check if the customer ID already exists and update the inquiry count
    for row in rows:
        if row[0] == str(customer_id):
            row[2] = str(int(row[2]) + 1)  # Increment the inquiry count
            customer_found = True
            break

    # If the customer was not found, add a new row with inquiry count starting from 0
    if not customer_found:
        rows.append([str(customer_id), customer_name, '0'])

    # Write the updated data back to the CSV file
    with open(csv_file_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerows(rows)

    # print(f"Customer data updated for ID {customer_id} with name {customer_name}.")

def print_prompt(prompt: str):
    print("*" * 80)
    print(f"Prompt: {prompt}")
    print("*" * 80)



# Main function to run the triage agent
def main():
    print(f"Pydantic version {pydantic.__version__}")
    print(f"Pydantic version {pydantic_ai.__version__}")
    deps = TriageDependencies(support_agent=support_agent, loan_agent=loan_agent, customer_id=123)
    prompt = 'What is my balance?'
    print_prompt(prompt)
    result = triage_agent.run_sync(prompt, deps=deps)
    print(result.data.text_response)
    # print(result.data.model_dump_json(indent=2))
    # """
    # {
    #   "department": "support",
    #   "response": {
    #     "support_advice": "Your current account balance is $123.45.",
    #     "block_card": false,
    #     "risk": 2,
    #     "customer_tracking_id": "13673e99-70ff-4851-8737-d06e66151234"
    #   },
    #   "text_response": "Your current account balance is $123.45."
    # }
    #     """
    #
    prompt = 'My card is lost. Please help!'
    print_prompt(prompt)
    result = triage_agent.run_sync(prompt, deps=deps)
    print(result.data.text_response)
    # print(result.data.model_dump_json(indent=2))
    # """
    # {
    #   "department": "support",
    #   "response": {
    #     "support_advice": "I'm sorry to hear that, John. We are temporarily blocking your card to prevent unauthorized transactions.",
    #     "block_card": true,
    #     "risk": 8,
    #     "customer_tracking_id": "04ee6c84-d996-43ae-b049-466c36249042"
    #   },
    #   "text_response": "I'm sorry to hear that, John. We are temporarily blocking your card to prevent unauthorized transactions."
    # }
    #     """
    prompt = 'What is the status of my loan?'
    print_prompt(prompt)
    result = triage_agent.run_sync(prompt, deps=deps)
    print(result.data.text_response)
    # print(result.data.model_dump_json(indent=2))

    """
    {
      "department": "loan",
      "response": {
        "loan_approval_status": "Active",
        "loan_balance": 5000.0,
        "customer_tracking_id": "3ec98579-43cc-4fb0-86eb-bd49ac66479c"
      },
      "text_response": "Your loan status is currently 'Active' and you have a remaining loan balance of $5000. If you need further assistance, feel free to reach out!"
    }
        """
    prompt = 'How tall is Eiffel tower ?'
    print_prompt(prompt)
    result = triage_agent.run_sync(prompt, deps=deps)
    print(result.data.text_response)
    #print(result.data.model_dump_json(indent=2))

if __name__ == '__main__':
    main()

