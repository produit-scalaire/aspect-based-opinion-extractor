from typing import Literal
from pydantic import BaseModel, Field

# LangChain imports
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser

# 1. Define the strict output schema using Pydantic
class OpinionSchema(BaseModel):
    Price: Literal["Positive", "Negative", "Mixed", "No Opinion"] = Field(
        description="Sentiment regarding the price. Must be exactly one of the literal values."
    )
    Food: Literal["Positive", "Negative", "Mixed", "No Opinion"] = Field(
        description="Sentiment regarding the food. Must be exactly one of the literal values."
    )
    Service: Literal["Positive", "Negative", "Mixed", "No Opinion"] = Field(
        description="Sentiment regarding the service. Must be exactly one of the literal values."
    )

class OpinionExtractor:
    # SET THE FOLLOWING CLASS VARIABLE to "FT" if you implemented a fine-tuning approach
    method: Literal["NOFT", "FT"] = "NOFT"

    # DO NOT MODIFY THE SIGNATURE OF THIS METHOD, add code to implement it
    def __init__(self, cfg) -> None:
        self.cfg = cfg
        self.target_model = self.cfg.ollama_model

        # Use OpenAI-compatible Ollama endpoint with authorized langchain-openai package.
        self.llm = ChatOpenAI(
            api_key="",
            base_url=self.cfg.ollama_url,
            model=self.target_model,
            temperature=0.0,
        )
        
        # Initialize the LangChain parser with our strict Pydantic schema
        self.parser = JsonOutputParser(pydantic_object=OpinionSchema)
        
        # Create a robust LangChain prompt template
        # {format_instructions} will be auto-populated by LangChain based on the Pydantic model
        self.prompt = ChatPromptTemplate.from_messages([
            ("system", "You are an expert NLP classifier for restaurant reviews.\n"
                       "You must extract the sentiment for 'Price', 'Food', and 'Service'.\n\n"
                       "{format_instructions}"),
            ("human", "Review: {review}")
        ])
        
        # Build the LangChain pipeline (LCEL - LangChain Expression Language)
        self.chain = self.prompt | self.llm | self.parser

        # Validation set for final safety check
        self.valid_labels = {"Positive", "Negative", "Mixed", "No Opinion"}

    # DO NOT MODIFY THE SIGNATURE OF THIS METHOD, add code to implement it
    def train(self, train_data: list[dict], val_data: list[dict]) -> None:
        """
        Trains the model, if OpinionExtractor.method=="FT"
        """
        pass

    # DO NOT MODIFY THE SIGNATURE OF THIS METHOD, add code to implement it
    def predict(self, texts: list[str]) -> list[dict]:
        """
        :param texts: list of reviews from which to extract the opinion values
        :return: a list of dicts, one per input review, containing the opinion values for the 3 aspects.
        """
        results = []
        
        # Prepare inputs with format instructions injected by LangChain
        format_instructions = self.parser.get_format_instructions()
        inputs = [{"review": text, "format_instructions": format_instructions} for text in texts]
        
        try:
            parsed_outputs = self.chain.batch(inputs, config={"max_concurrency": 2})
            
            # Map outputs securely to avoid any hallucinated keys
            for out in parsed_outputs:
                results.append({
                    "Price": out.get("Price") if out.get("Price") in self.valid_labels else "No Opinion",
                    "Food": out.get("Food") if out.get("Food") in self.valid_labels else "No Opinion",
                    "Service": out.get("Service") if out.get("Service") in self.valid_labels else "No Opinion"
                })
                
        except Exception as e:
            print(f"\n[LangChain Batch Error]: {type(e).__name__} - {e}")
            # Fallback to empty predictions if the whole batch fails
            results = [{"Price": "No Opinion", "Food": "No Opinion", "Service": "No Opinion"} for _ in texts]
            
        return results