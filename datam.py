from pymongo import MongoClient
from pydantic import BaseModel, validator
from datetime import datetime
import re

# MongoDB setup
client = MongoClient('localhost', 27017)
db = client['test-database']
collection = db['test-collection']

# Setup schema with Pydantic
class Freshness(BaseModel):
    timestamp: datetime
    produce: str
    freshness: float
    expected_life_span: str

    # Set timestamp if it's not provided
    @validator('timestamp', pre=True, always=True)
    def set_timestamp(cls, v):
        return v or datetime.utcnow()

# Extract details function
def extract_details(class_name):
    try:
        # Regex pattern to extract the produce name and the shelf life range (e.g., "5-10")
        match = re.match(r"([a-zA-Z]+)\((\d+)-(\d+)\)", class_name)
        
        if match:
            produce = match.group(1)  # Extract the produce (e.g., "Banana")
            lower_value = int(match.group(2))  # Extract the lower value of the shelf life (e.g., 5)
            upper_value = int(match.group(3))  # Extract the upper value of the shelf life (e.g., 10)
            
            # Calculate the freshness as the lower value divided by 5
            freshness = lower_value / 5
            
            # Create a Freshness Pydantic model instance
            freshness_data = Freshness(
                produce=produce,
                expected_life_span=f"{lower_value}-{upper_value}",
                freshness=freshness,
                timestamp=datetime.utcnow()  # Set the current timestamp
            )

            # Print for debugging purposes
            print(f"Produce: {produce}, Shelf Life: {lower_value}-{upper_value}, Freshness: {freshness}")

            # Insert the Freshness data into MongoDB
            collection.insert_one(freshness_data.dict())  # Use .dict() to convert Pydantic model to a dictionary

            return freshness_data.dict()  # Return the dictionary form of the model for further processing
        else:
            raise ValueError(f"Invalid class_name format: {class_name}")
    
    except Exception as e:
        print(f"Error in extract_details: {str(e)}")
        raise  # Re-raise the error to propagate it to the calling function

# Example usage
class_name = "Banana(5-10)"
try:
    demore = extract_details(class_name)  # Extract details and insert into MongoDB
    print("Data inserted successfully:", demore)  # Print the inserted data
except Exception as e:
    print(f"Failed to extract and insert data: {str(e)}")
