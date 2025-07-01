from httpx import Timeout
from mostlyai import mock
import litellm

class MyCustomLLM(litellm.CustomLLM):
    def completion(self, *args, **kwargs) -> litellm.ModelResponse:
        return litellm.completion(*args, **kwargs)  # type: ignore
    
    async def acompletion(self, *args, **kwargs) -> litellm.Coroutine[litellm.Any, litellm.Any, litellm.ModelResponse]:
        return litellm.acompletion(*args, **kwargs)

my_custom_llm = MyCustomLLM()

litellm.custom_provider_map = [ # 👈 KEY STEP - REGISTER HANDLER
        {"provider": "my-custom-llm", "custom_handler": my_custom_llm}
    ]


tables = {
    "guests": {
        "prompt": "Guests of an Alpine ski hotel in Austria",
        "columns": {
            "nationality": {"prompt": "2-letter code for the nationality", "dtype": "string"},
            "name": {"prompt": "first name and last name of the guest", "dtype": "string"},
            "gender": {"dtype": "category", "values": ["male", "female"]},
            "age": {"prompt": "age in years; min: 18, max: 80; avg: 25", "dtype": "integer"},
            "date_of_birth": {"prompt": "date of birth", "dtype": "date"},
            "checkin_time": {"prompt": "the check in timestamp of the guest; may 2025", "dtype": "datetime"},
            "is_vip": {"prompt": "is the guest a VIP", "dtype": "boolean"},
            "price_per_night": {"prompt": "price paid per night, in EUR", "dtype": "float"},
            "room_number": {"prompt": "room number", "dtype": "integer", "values": [101, 102, 103, 201, 202, 203, 204]}
        },
    }
}
df = mock.sample(
    tables=tables,   # provide table and column definitions
    sample_size=10,  # generate 10 records
    # model="openai/gpt-4.1-nano",  # select the LLM model (optional)
    model="my-custom-llm/gpt-4.1-nano",
)
print(df)